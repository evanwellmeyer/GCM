# SCM teaching notebooks

These notebooks introduce the GCM repository's single-column atmosphere model. They use the default 20-level, conservative mass-flux configuration and its saved equilibrium reference state.

1. [`01_meet_the_column.ipynb`](01_meet_the_column.ipynb) introduces the vertical grid and each physics parameterization separately.
2. [`02_experiments.ipynb`](02_experiments.ipynb) combines the physics into experiments with atmospheric tendencies, radiation, convection, clouds, and prescribed dynamical forcing.

The notebooks can run in Google Colab using their launch badges. For local use, open the repository in VS Code or Jupyter and select a Python environment containing the project and notebook dependencies:

```bash
conda activate atm407
cd /Users/evanwellmeyer/Documents/GCM
jupyter lab
```

Run the notebooks in order. Both load `scm/configs/atm407.toml` and the canonical files in `notebooks/data/`; do not substitute an experimental checkpoint unless the assignment specifically asks for it.

For developers: `scm/configs/default.toml` now aliases this teaching setup; the UW/BOMEX candidate remains a separate experiment. See the [configuration inventory](../scm/configs/README.md) and [active code map](../docs/column_code_map.md). Passing equilibrium criteria does not remove the documented cloud-profile limitations.

## How to work through the notebooks

Read the text before each code cell, then run the cells one at a time from top to bottom. Many sections ask you to predict what will happen before running an experiment. Write down your prediction first, even if you are unsure, and then compare it with the model output.

Enter your responses in the blanks provided or add a Markdown cell directly below the question. Record both what happened and what you think it means physically. When a slider or parameter is available, begin with the default value, change one quantity at a time, and note the values used for any result you discuss. A useful response should distinguish among:

- your prediction;
- your observation from the figures or printed diagnostics;
- your explanation of the physical process; and
- what the experiment taught you or left unresolved.

If a cell reports an error, first confirm that all preceding setup cells have run successfully. Restart the kernel and run from the beginning if variables or figures appear inconsistent with the written instructions.
