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
