# 24-788 Mini-Project: HOMO-LUMO Gap Prediction on QM9

**Course:** 24-788 Introduction to Deep Learning — Spring 2026  
**Task:** Predict the HOMO-LUMO energy gap (Δε) for small organic molecules  
**Dataset:** QM9 (~130 k molecules with DFT-computed quantum properties)  
**Metric:** Mean Absolute Error (MAE) in meV

## Models

| Model | Description | Reference |
|---|---|---|
| Baseline — GCN | Graph Convolutional Network | Kipf & Welling, ICLR 2017 |
| Variant 1 — Dendritic GCN | GCN augmented with learnable dendritic branches via Perforated AI | Beniaguev et al., Nature Communications 2025 |
| Variant 2 — Sparse GCN | GCN trained with RigL dynamic sparse training (50% sparsity) | Evci et al., ICML 2020 |

## AI Tool Use

Code development was assisted by Claude (Anthropic). All analysis, written results, and interpretation are the author's own work.

---

## Environment Setup

All experiments are designed to run on **Google Colab (free tier)** with a T4 GPU.

### Option A — Run in Colab (recommended)

1. Upload `qm9_mini_project.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Run all cells top-to-bottom. The notebook handles all installs and data downloads automatically.

### Option B — Run locally

**Requirements:** Python 3.9+, CUDA-capable GPU (recommended)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install rigl-torch
pip install perforatedai
```

Then open `qm9_mini_project.ipynb` in Jupyter and run all cells.

### Data Download

QM9 is downloaded automatically by PyTorch Geometric on first run (~300 MB). No manual download is needed. The dataset is cached to `qm9_project/data/` inside your Google Drive (or `/content/qm9_project/data/` locally).

---

## Reproducing Key Results

### After a completed training run

The notebook saves three checkpoint files and three history JSON files to `qm9_project/checkpoints/` in your Google Drive:

```
qm9_project/
  checkpoints/
    baseline_best.pt       # best Baseline GCN weights
    dendritic_best.pt      # best Dendritic GCN weights
    sparse_best.pt         # best Sparse GCN weights
  hist_baseline.json       # per-epoch train loss and val MAE
  hist_dendritic.json
  hist_sparse.json
  results.json             # final test MAEs for all three models
  results_comparison.pdf   # figures used in the report
```

### To regenerate metrics and figures without retraining

1. Open `qm9_mini_project.ipynb` in Colab.
2. Run **cells 1–5** only (Drive mount, installs, imports, config, data).
3. Jump to **Section 10 — Reproduce Results** and run cells 32 and 33.

Cell 32 loads all three checkpoints and prints the test MAE table.  
Cell 33 loads the history JSON files and regenerates `results_comparison.pdf`.

No training is required. Expected runtime for the reproduce section: ~2 minutes on a T4 GPU.

---

## Expected Results

Approximate test MAEs after 50 epochs (results will vary slightly by run):

| Model | Test MAE (meV) |
|---|---|
| Baseline GCN | ~120–160 |
| Dendritic GCN | varies (PAI restructures iteratively) |
| Sparse GCN (RigL) | comparable to baseline at 50% sparsity |
| SotA GNN reference | ~50 |

Note: the goal of this project is not to match SotA but to compare the three architectures fairly and analyse why they behave differently.

---

## Repository Structure

```
qm9_mini_project.ipynb   # main Colab notebook (training + evaluation)
README.md                # this file
repro_eval.py            # source for Section 10a (reproduce metrics)
repro_figures.py         # source for Section 10b (reproduce figures)
make_notebook.py         # script used to generate the .ipynb
```

---

## References

- Kipf & Welling, "Semi-supervised Classification with Graph Convolutional Networks," ICLR 2017.
- Beniaguev et al., "Dendrites endow artificial neural networks with accurate, robust and parameter-efficient learning," Nature Communications, 2025.
- Evci et al., "Rigging the Lottery: Making All Tickets Winners," ICML 2020.
- Ramakrishnan et al., "Quantum chemistry structures and properties of 134 kilo molecules," Scientific Data, 2014 (QM9 dataset).
