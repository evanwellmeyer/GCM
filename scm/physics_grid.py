"""Conservative vertical refinement for column physics.

The host model owns the atmospheric grid. Parameterizations that need more
vertical detail can use a refined pressure grid and return their tendencies to
the host without changing the host state layout.
"""

from dataclasses import dataclass

import torch

from scm.thermo import (
    dp_from_ps,
    grid_from_pressure_interfaces,
    half_level_coordinate,
    pressure_at_full,
    pressure_at_half,
)


_state_fields = (
    "t",
    "q",
    "qc",
    "u",
    "v",
    "tke",
    "cloud_fraction",
)


def _as_batched_interfaces(interfaces, *, device=None, dtype=None):
    values = torch.as_tensor(interfaces, device=device, dtype=dtype)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("pressure interfaces must have shape (n + 1,) or (batch, n + 1)")
    if torch.any(values[:, 1:] <= values[:, :-1]):
        raise ValueError("pressure interfaces must increase from top to bottom")
    return values


def _broadcast_batches(first, second):
    batch = max(first.shape[0], second.shape[0])
    if first.shape[0] not in (1, batch) or second.shape[0] not in (1, batch):
        raise ValueError("pressure-interface batches are incompatible")
    return first.expand(batch, -1), second.expand(batch, -1)


def layer_overlap(source_interfaces, target_interfaces):
    """Return pressure overlap with shape ``(batch, target, source)``."""

    source = _as_batched_interfaces(source_interfaces)
    target = _as_batched_interfaces(
        target_interfaces,
        device=source.device,
        dtype=source.dtype,
    )
    source, target = _broadcast_batches(source, target)

    source_top = source[:, :-1].unsqueeze(1)
    source_bottom = source[:, 1:].unsqueeze(1)
    target_top = target[:, :-1].unsqueeze(2)
    target_bottom = target[:, 1:].unsqueeze(2)
    return (
        torch.minimum(source_bottom, target_bottom)
        - torch.maximum(source_top, target_top)
    ).clamp(min=0.0)


def conservative_remap(values, source_interfaces, target_interfaces):
    """Remap layer means while preserving their pressure integral.

    ``values`` must have shape ``(batch, source_layers)``. The remap assumes a
    piecewise-constant value inside each source layer. Source and target grids
    must cover the same pressure column.
    """

    values = torch.as_tensor(values)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError("layer values must have shape (layers,) or (batch, layers)")

    source = _as_batched_interfaces(
        source_interfaces,
        device=values.device,
        dtype=values.dtype,
    )
    target = _as_batched_interfaces(
        target_interfaces,
        device=values.device,
        dtype=values.dtype,
    )
    source, target = _broadcast_batches(source, target)
    batch = max(values.shape[0], source.shape[0])
    if values.shape[0] not in (1, batch) or source.shape[0] not in (1, batch):
        raise ValueError("layer-value and pressure-interface batches are incompatible")
    values = values.expand(batch, -1)
    source = source.expand(batch, -1)
    target = target.expand(batch, -1)

    if values.shape[1] != source.shape[1] - 1:
        raise ValueError("the number of values must match the source layers")
    if not torch.allclose(source[:, (0, -1)], target[:, (0, -1)]):
        raise ValueError("source and target grids must cover the same pressure column")

    overlap = layer_overlap(source, target)
    target_depth = target[:, 1:] - target[:, :-1]
    return torch.einsum("bts,bs->bt", overlap, values) / target_depth


def _refinement_counts(coordinate, sublevels, top):
    if sublevels < 1:
        raise ValueError("physics-grid sublevels must be at least one")
    reference = coordinate[0] if coordinate.ndim == 2 else coordinate
    if coordinate.ndim == 2 and not torch.allclose(
        coordinate,
        reference.unsqueeze(0).expand_as(coordinate),
    ):
        raise ValueError("all columns must use the same vertical-coordinate layout")
    lower_interface = reference[1:]
    refined = lower_interface > float(top)
    return torch.where(
        refined,
        torch.full_like(lower_interface, int(sublevels), dtype=torch.long),
        torch.ones_like(lower_interface, dtype=torch.long),
    )


def _refine_interfaces(interfaces, counts):
    pieces = [interfaces[:, :1]]
    parents = []
    for layer, count in enumerate(counts.tolist()):
        fraction = torch.arange(
            1,
            count + 1,
            device=interfaces.device,
            dtype=interfaces.dtype,
        ) / count
        top = interfaces[:, layer:layer + 1]
        bottom = interfaces[:, layer + 1:layer + 2]
        pieces.append(top + (bottom - top) * fraction.unsqueeze(0))
        parents.extend([layer] * count)
    return torch.cat(pieces, dim=1), torch.tensor(parents, device=interfaces.device)


@dataclass(frozen=True)
class PhysicsGrid:
    """A refined grid and the conservative map connecting it to the host."""

    host_interfaces: torch.Tensor
    physics_interfaces: torch.Tensor
    grid: dict
    parent_layer: torch.Tensor

    def to_physics(self, values):
        return conservative_remap(
            values,
            self.host_interfaces,
            self.physics_interfaces,
        )

    def to_host(self, values):
        return conservative_remap(
            values,
            self.physics_interfaces,
            self.host_interfaces,
        )

    def state_to_physics(self, state):
        """Copy the column state and refine the intensive atmospheric fields."""

        refined = dict(state)
        for name in _state_fields:
            if name in state:
                refined[name] = self.to_physics(state[name])
        refined["p"] = pressure_at_full(self.grid, state["ps"])
        refined["dp"] = dp_from_ps(self.grid, state["ps"])
        return refined

    def output_to_host(self, output):
        """Return layer-shaped scheme outputs to the host grid conservatively."""

        restored = dict(output)
        physics_levels = self.grid["nlevels"]
        for name, values in output.items():
            if (
                torch.is_tensor(values)
                and values.ndim == 2
                and values.shape[1] == physics_levels
            ):
                restored[name] = self.to_host(values)
        return restored


def make_physics_grid(host_grid, ps, *, sublevels=4, top=0.70):
    """Refine host layers below a dimensionless vertical-coordinate threshold."""

    host_interfaces = pressure_at_half(host_grid, ps)
    coordinate = half_level_coordinate(
        host_grid,
        ps=ps,
        batch=host_interfaces.shape[0],
        device=host_interfaces.device,
        dtype=host_interfaces.dtype,
    )
    counts = _refinement_counts(coordinate, int(sublevels), float(top))
    physics_interfaces, parent_layer = _refine_interfaces(host_interfaces, counts)
    physics_coordinate, _ = _refine_interfaces(coordinate, counts)

    grid = grid_from_pressure_interfaces(
        physics_interfaces,
        device=physics_interfaces.device,
        dtype=physics_interfaces.dtype,
    )
    grid["sigma_half"] = physics_coordinate
    grid["sigma_full"] = 0.5 * (
        physics_coordinate[:, :-1] + physics_coordinate[:, 1:]
    )
    grid["dsigma"] = physics_coordinate[:, 1:] - physics_coordinate[:, :-1]
    grid["host_parent_layer"] = parent_layer
    return PhysicsGrid(
        host_interfaces=host_interfaces,
        physics_interfaces=physics_interfaces,
        grid=grid,
        parent_layer=parent_layer,
    )
