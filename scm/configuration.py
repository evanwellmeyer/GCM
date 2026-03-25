from copy import deepcopy
from pathlib import Path
import tomllib


CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.toml"


def _non_null_items(mapping):
    return {k: v for k, v in mapping.items() if v is not None}


def deep_merge(base, override):
    """recursively merge override into base and return the merged dict."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_run_config(path=None):
    """load the default SCM config, optionally merged with a user config."""

    with DEFAULT_CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)

    if path is None:
        return config

    user_path = Path(path)
    with user_path.open("rb") as f:
        override = tomllib.load(f)

    config = deep_merge(config, override)
    config["_config_path"] = str(user_path)
    return config


def derive_radiation_mode(trace_gases_enabled, clouds_enabled):
    if trace_gases_enabled and clouds_enabled:
        return "semi_gray_plus_trace_gases_clouds"
    if trace_gases_enabled:
        return "semi_gray_plus_trace_gases"
    if clouds_enabled:
        return "semi_gray_plus_clouds"
    return "semi_gray"


def extract_param_overrides(config):
    """flatten structured config sections into the params dict expected by the SCM."""

    params = dict(config.get("params", {}))
    radiation = config.get("radiation", {})
    if radiation:
        params.update(_non_null_items({
            "radiation_scheme": radiation.get("scheme"),
            "radiation_mode": radiation.get("mode"),
        }))

        params.update(_non_null_items(radiation.get("longwave", {})))
        params.update(_non_null_items(radiation.get("shortwave", {})))

        trace = radiation.get("trace_gases", {})
        if trace:
            trace_enabled = bool(trace.get("enabled", False))
            params["trace_gases_enabled"] = trace_enabled
            params.update(_non_null_items({
                "ch4": trace.get("ch4"),
                "ch4_ref": trace.get("ch4_ref"),
                "ch4_base_tau": trace.get("ch4_base_tau"),
                "ch4_log_factor": trace.get("ch4_log_factor"),
                "n2o": trace.get("n2o"),
                "n2o_ref": trace.get("n2o_ref"),
                "n2o_base_tau": trace.get("n2o_base_tau"),
                "n2o_log_factor": trace.get("n2o_log_factor"),
                "o3_lw_tau": trace.get("o3_lw_tau"),
                "o3_sw_tau": trace.get("o3_sw_tau"),
                "other_ghg_tau": trace.get("other_ghg_tau"),
            }))
        else:
            trace_enabled = False

        clouds = radiation.get("clouds", {})
        if clouds:
            clouds_enabled = bool(clouds.get("enabled", False))
            params["cloud_radiative_effects_enabled"] = clouds_enabled
            params.update(_non_null_items({
                "cloud_fraction": clouds.get("cloud_fraction"),
                "cloud_sw_reflectivity": clouds.get("cloud_sw_reflectivity"),
                "cloud_sw_tau": clouds.get("cloud_sw_tau"),
                "cloud_lw_tau": clouds.get("cloud_lw_tau"),
                "cloud_top_sigma": clouds.get("cloud_top_sigma"),
                "cloud_bottom_sigma": clouds.get("cloud_bottom_sigma"),
            }))
        else:
            clouds_enabled = False

        if "radiation_mode" not in params:
            params["radiation_mode"] = derive_radiation_mode(trace_enabled, clouds_enabled)

    return params
