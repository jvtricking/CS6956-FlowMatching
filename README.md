# Conditional Flow Matching — Liver Laboratory Data

**Goal:** Learn a conditional generative model for eight liver-related blood-test measurements, given the UCI ILPD liver-patient vs non-liver-patient label. Compare network width/depth, Euler sampling steps, and a conditional Gaussian baseline. Results are for a probabilistic deep learning class, **not medical diagnosis**.

**Data:** [UCI ILPD dataset 225](https://archive.ics.uci.edu/dataset/225/ilpd) (583 patient records, 416 liver-patient and 167 non-liver-patient). Data downloads automatically during the first notebook run (internet required once). A separate dataset file is not shipped. Citation: Ramana, B. & Venkateswarlu, N. (2022), *ILPD*, UCI Machine Learning Repository, DOI [10.24432/C5D02C](https://doi.org/10.24432/C5D02C). CC BY 4.0.

## Run in VS Code (Windows PowerShell)

1. Open this folder in VS Code and install the **Python** and **Jupyter** extensions.
2. Create and activate a local environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Open `liver_flow_matching.ipynb` and select the `.venv` Python kernel in the top right.
4. Run all cells from top to bottom. Results are saved to `outputs/` (PNG plots and CSV tables).
5. Edit `.env` to change training settings; restart kernel and rerun. Set `RUN_EPOCH_EXPERIMENT=1` to test additional training durations.

**macOS/Linux:** Create with `python3 -m venv .venv`, activate using `source .venv/bin/activate`, then use the same `pip` commands.

**Performance:** CPU compatible; PyTorch CUDA automatically used if installed with CUDA support. Start with `EPOCHS=100` and `TOY_EPOCHS=150` if you want a quicker initial check.

## What to discuss in the report

- Toy 2D generation and conditional flow-matching objective.
- Real-data preprocessing, train/test split, label imbalance and dataset biases.
- Architecture comparisons (32x2, 64x2, 64x4) versus conditional Gaussian.
- Per-label holdout fidelity: MMD², Wasserstein, correlation error, and basic physical consistency checks.
- Tradeoff between Euler sample steps and generation time.
- Data constraints: tiny dataset, possible memorization, and no clinical interpretation.

**Caution:** Flow matching uses straight-line conditional interpolation paths **during training**, but learned ODE trajectories are not guaranteed to be straight. Generative performance is not diagnostic performance. The notebook has been syntax-checked, but full training has not been run in this environment.
