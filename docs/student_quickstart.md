# Student Quickstart

## Recommended: Google Colab

No local installation is required. Open the setup notebook, choose
**Runtime → Run all**, and confirm that all tests pass:

[Open `01_setup.ipynb` in Colab](https://colab.research.google.com/github/evanwellmeyer/GCM/blob/main/notebooks/01_setup.ipynb)

Then open the exercise notebook and again choose **Runtime → Run all**:

[Open `02_experiments.ipynb` in Colab](https://colab.research.google.com/github/evanwellmeyer/GCM/blob/main/notebooks/02_experiments.ipynb)

For the senior atmospheric dynamics version, open the ATM 407 lab instead:

[Open `02_experiments_atm407.ipynb` in Colab](https://colab.research.google.com/github/evanwellmeyer/GCM/blob/main/notebooks/02_experiments_atm407.ipynb)

Each notebook installs the model into its current Colab runtime. Colab storage is
temporary, so students should download their completed notebook or save a copy to
Google Drive before closing the session.

## Local installation

This guide gets you from a fresh clone to a tested SCM and a short experiment. Run every command from the repository root.

## 1. Create an isolated Python environment

The SCM supports Python 3.11 and newer. A virtual environment keeps its packages separate from the rest of your computer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate the environment with `.venv\\Scripts\\Activate.ps1` instead.

## 2. Install the SCM

Install the model, plotting tools, and test tools in editable mode:

```bash
python -m pip install -e '.[dev]'
```

Editable mode means changes you make to the source code take effect without reinstalling the package.

PyTorch wheels vary by operating system and accelerator. The command above installs the default wheel for your platform. If your course or lab provides a CUDA-specific PyTorch command, run that command first and then install the SCM.

## 3. Verify the installation

```bash
python -m pytest
python -c "import scm, torch; print('SCM import passed; PyTorch', torch.__version__)"
```

The test command intentionally checks only the standalone SCM. The nested `VFS` and paper directories are separate work and are not part of this package.

## 4. Run a short experiment

```bash
scm-run --demo --device cpu --no-plot
```

The demo is short relative to paper experiments, but it can still take several minutes on a laptop. Output files are written below `outputs/column/`.

To include a diagnostic figure, omit `--no-plot`:

```bash
scm-run --demo --device cpu
```

## 5. Read and change one configuration

Start with `scm/configs/default.toml`. Copy it before making changes so the repository default remains a useful reference:

```bash
cp scm/configs/default.toml my_experiment.toml
scm-run --config my_experiment.toml --device cpu --no-plot
```

Change one setting at a time, record the exact configuration with each result, and check conservation and equilibrium diagnostics before interpreting a run scientifically.

## Paper-run checklist

Before starting a long or published experiment:

1. Record the Git commit with `git rev-parse HEAD` and keep the run configuration.
2. Run `python -m pytest` in the same environment used for the experiment.
3. Begin with the frozen `scm/configs/mf_baseline_v1.toml` reference unless your experiment requires a documented alternative.
4. Use a new output location or archive earlier output before rerunning the same case.
5. Inspect energy-budget, equilibrium, and forcing diagnostics; a completed process is not automatically a scientifically valid run.
6. Record Python, PyTorch, hardware, configuration, random seed, and wall-clock time in the experiment log.

For model design, configuration details, benchmark commands, and longer runs, continue with the main [README](../README.md).

## Notebook course path

Students who are using Jupyter can instead begin with
[`01_setup.ipynb`](../notebooks/01_setup.ipynb). After installation and kernel
verification, [`02_experiments.ipynb`](../notebooks/02_experiments.ipynb)
provides self-contained exercises without requiring students to edit TOML files.
