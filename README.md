# Conditional Flow Matching — Liver Laboratory Data

**Goal:** Learn a conditional generative model for eight liver-related blood-test measurements, given the UCI ILPD liver-patient vs non-liver-patient label. Compare network width/depth, Euler sampling steps, and a conditional Gaussian baseline. Results are for a probabilistic deep learning class, **not medical diagnosis**.

**Data:** [UCI ILPD dataset 225](https://archive.ics.uci.edu/dataset/225/ilpd) (583 patient records, 416 liver-patient and 167 non-liver-patient). Data downloads automatically during the first notebook run (internet required once). A separate dataset file is not shipped. Citation: Ramana, B. & Venkateswarlu, N. (2022), *ILPD*, UCI Machine Learning Repository, DOI [10.24432/C5D02C](https://doi.org/10.24432/C5D02C). CC BY 4.0.

## Run in VS Code (Windows PowerShell / Conda)

1. Install Miniconda or Anaconda, and install the **Python** and **Jupyter** extensions in VS Code.
2. Open this project folder in VS Code, then create the Conda environment in a terminal:
   ```powershell
   conda env create -f environments.yml
   conda activate liver-flow-matching
   ```
   If the environment already exists and you change `environments.yml`, update it with:
   ```powershell
   conda env update -f environments.yml --prune
   ```
3. Open `liver_flow_matching.ipynb`. In the upper-right corner choose **Select Kernel** → **Python Environments** → `liver-flow-matching` (or the matching Conda Python interpreter).
4. Run all cells from top to bottom. Figures and tables are saved to `outputs/`.
5. Change the configuration variables near the top of the notebook (`EPOCHS`, `TOY_EPOCHS`, etc.) to adjust experiments. Enable the optional training-duration experiment by setting `RUN_EPOCH_EXPERIMENT = True` in its code cell.

**File roles:** `environments.yml` creates the Conda environment; experiment settings live directly in the notebook. `requirements.txt` is an optional pip alternative.

**Performance:** This environment uses **PyTorch 2.5.1 with CUDA 12.4** and pip-built numerical packages to avoid mixing Intel MKL/OpenMP with PyTorch's native Windows libraries. A compatible NVIDIA GPU and driver are required; the notebook automatically falls back to CPU if CUDA is unavailable. For a quick first run, set `EPOCHS = 100` and `TOY_EPOCHS = 150` in the notebook configuration cell.

## What to discuss in the report

- Toy 2D generation and conditional flow-matching objective.
- Real-data preprocessing, train/test split, label imbalance and dataset biases.
- Architecture comparisons (32x2, 64x2, 64x4) versus conditional Gaussian.
- Per-label holdout fidelity: MMD², Wasserstein, correlation error, and basic physical consistency checks.
- Tradeoff between Euler sample steps and generation time.
- Data constraints: tiny dataset, possible memorization, and no clinical interpretation.

**Caution:** Flow matching uses straight-line conditional interpolation paths **during training**, but learned ODE trajectories are not guaranteed to be straight. Generative performance is not diagnostic performance. The notebook has been syntax-checked, but full training has not been run in this environment.
