"""
Generates qm9_mini_project.ipynb — a Google Colab notebook for 24-788 mini-project.

Models:
  1. Baseline  : GCN (Kipf & Welling, ICLR 2017)
  2. Variant 1 : Dendritic GCN via Perforated AI (Nature Comms 2025)
  3. Variant 2 : Sparse GCN via RigL (Evci et al., ICML 2020)

Task: HOMO-LUMO gap prediction on QM9.
"""

import json, os

OUT = "/sessions/fervent-funny-heisenberg/mnt/outputs/qm9_mini_project.ipynb"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

# ===========================================================================
# 0. TITLE
# ===========================================================================
cells.append(md(
"""# 24-788 Mini-Project: HOMO-LUMO Gap Prediction on QM9

**Task:** Predict the HOMO-LUMO energy gap (Δε) for small organic molecules.
**Dataset:** QM9 — ~130 k molecules with DFT-computed quantum properties.
**Metric:** Mean Absolute Error (MAE) in meV.

| Model | Reference |
|---|---|
| Baseline — Graph Convolutional Network (GCN) | Kipf & Welling, ICLR 2017 |
| Variant 1 — Dendritic GCN (Perforated AI) | Beniaguev et al., Nature Comms 2025 |
| Variant 2 — Sparse GCN (RigL) | Evci et al., ICML 2020 |"""
))

# ===========================================================================
# 1. GOOGLE DRIVE MOUNT  (saves checkpoints so free Colab timeouts don't hurt)
# ===========================================================================
cells.append(md("## 1 · Setup"))

cells.append(code(
"""# Mount Google Drive so checkpoints persist across sessions.
# If Drive is unavailable, checkpoints are saved locally (/content/checkpoints).
import os

USE_DRIVE = True   # set False to skip Drive mount

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        SAVE_DIR = '/content/drive/MyDrive/qm9_project'
    except Exception:
        USE_DRIVE = False

if not USE_DRIVE:
    SAVE_DIR = '/content/qm9_project'

os.makedirs(SAVE_DIR, exist_ok=True)
CKPT_DIR = os.path.join(SAVE_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Save directory: {SAVE_DIR}")"""
))

# ---------------------------------------------------------------------------
cells.append(code(
"""# Install dependencies (run once per Colab session).
# torch-geometric: graph ML library
# rigl-torch     : RigL sparse training scheduler
# perforatedai   : Perforated AI dendritic augmentation
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

pip('torch-geometric')
pip('rigl-torch')
pip('perforatedai')
print("All packages installed.")"""
))

# ---------------------------------------------------------------------------
cells.append(code(
"""import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import matplotlib.pyplot as plt
import json, os
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
torch.manual_seed(42)
np.random.seed(42)"""
))

# ---------------------------------------------------------------------------
# TensorBoard setup  (log dir lives inside SAVE_DIR → persists on Drive)
# ---------------------------------------------------------------------------
cells.append(code(
"""# TensorBoard — load extension and point it at the log directory.
# Logs persist in Google Drive across sessions.
%load_ext tensorboard

LOG_DIR = os.path.join(SAVE_DIR, 'runs')
os.makedirs(LOG_DIR, exist_ok=True)
print(f"TensorBoard log directory: {LOG_DIR}")"""
))

# ===========================================================================
# 2. CONFIG
# ===========================================================================
cells.append(md("## 2 · Configuration"))

cells.append(code(
"""# ---------- Target ----------
TARGET_IDX  = 4          # HOMO-LUMO gap (Δε), index in QM9.y — units: eV
TARGET_NAME = 'HOMO-LUMO gap'
TARGET_UNIT = 'eV'

# ---------- Architecture ----------
IN_CHANNELS = 11         # QM9 node features (one-hot atom type + valence info)
HIDDEN_DIM  = 128
NUM_LAYERS  = 4

# ---------- Training ----------
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS      = 50         # per model; free T4 handles ~3 min/epoch → ~2.5 h total
PATIENCE    = 12         # early stopping patience (val MAE)
LR_PATIENCE = 6          # ReduceLROnPlateau patience

# ---------- Splits (standard QM9) ----------
N_TRAIN = 110_000
N_VAL   =  10_000
# test = remainder (~10 k)

# ---------- RigL ----------
RIGL_SPARSITY = 0.5      # 50% of weights are zero
RIGL_DELTA    = 100      # topology update interval (steps)
RIGL_ALPHA    = 0.3      # fraction of weights redistributed each cycle

print("Configuration set.")"""
))

# ===========================================================================
# 3. DATA
# ===========================================================================
cells.append(md("## 3 · Data"))

cells.append(code(
"""# Download QM9 (~300 MB) — PyG handles this automatically.
dataset = QM9(root=os.path.join(SAVE_DIR, 'data'))
print(f"Dataset size : {len(dataset):,} molecules")
print(f"Node features: {dataset.num_node_features}")
print(f"Targets      : {dataset.data.y.shape[1]} properties per molecule")"""
))

cells.append(code(
"""# Reproducible shuffle → fixed train / val / test split.
generator = torch.Generator().manual_seed(42)
perm = torch.randperm(len(dataset), generator=generator)

train_idx = perm[:N_TRAIN]
val_idx   = perm[N_TRAIN : N_TRAIN + N_VAL]
test_idx  = perm[N_TRAIN + N_VAL :]

# Normalisation statistics computed from TRAINING SET ONLY.
# Units: eV → we store mean/std in eV and convert predictions to meV for reporting.
all_y    = dataset.data.y                  # [N, 19]
train_y  = all_y[train_idx, TARGET_IDX]
MEAN     = train_y.mean().item()
STD      = train_y.std().item()

print(f"Target   : {TARGET_NAME}")
print(f"Train set: mean = {MEAN:.4f} {TARGET_UNIT}, std = {STD:.4f} {TARGET_UNIT}")
print(f"Train / Val / Test: {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")"""
))

cells.append(code(
"""train_dataset = dataset[train_idx]
val_dataset   = dataset[val_idx]
test_dataset  = dataset[test_idx]

# num_workers=2 speeds up loading on Colab; reduce to 0 if you hit errors.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Batches per epoch: {len(train_loader):,}")"""
))

# ===========================================================================
# 4. MODEL
# ===========================================================================
cells.append(md(
"""## 4 · Baseline Model: Graph Convolutional Network (GCN)

Kipf & Welling, *"Semi-supervised Classification with Graph Convolutional Networks"*, ICLR 2017.

GCN propagates node features by averaging neighbour representations at each layer.
Molecules are naturally represented as graphs (atoms = nodes, bonds = edges), making GCN
a principled baseline. We aggregate node embeddings with global mean pooling and pass them
through a small MLP to produce a scalar energy prediction."""
))

cells.append(code(
"""class GCN(nn.Module):
    \"\"\"
    Multi-layer GCN for graph-level regression.

    Architecture:
        [GCNConv → BatchNorm → ReLU → Dropout] × num_layers
        → GlobalMeanPool
        → Linear(hidden, hidden//2) → ReLU → Dropout
        → Linear(hidden//2, 1)
    \"\"\"
    def __init__(self,
                 in_channels=IN_CHANNELS,
                 hidden=HIDDEN_DIM,
                 num_layers=NUM_LAYERS,
                 dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden, add_self_loops=True))
        self.bns.append(nn.BatchNorm1d(hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden, add_self_loops=True))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        x = x.float()
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)       # [num_graphs, hidden]
        return self.head(x).squeeze(-1)       # [num_graphs]

_tmp = GCN()
n_params = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
print(f"GCN parameters: {n_params:,}")
del _tmp"""
))

# ===========================================================================
# 5. TRAINING UTILITIES
# ===========================================================================
cells.append(md("## 5 · Training Utilities"))

cells.append(code(
"""def train_epoch(model, loader, optimizer, device,
               mean=None, std=None, target=TARGET_IDX):
    \"\"\"One training epoch. Targets are normalised before computing MSE loss.\"\"\"
    mean = mean if mean is not None else MEAN
    std  = std  if std  is not None else STD
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        y_norm = (data.y[:, target] - mean) / std    # normalise
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out, y_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, mean=None, std=None, target=TARGET_IDX):
    \"\"\"Returns MAE in original eV units (multiply by 1000 for meV).\"\"\"
    mean = mean if mean is not None else MEAN
    std  = std  if std  is not None else STD
    model.eval()
    mae_sum = 0.0
    for data in loader:
        data = data.to(device)
        out  = model(data.x, data.edge_index, data.batch)
        pred = out * std + mean                       # denormalise → eV
        true = data.y[:, target]
        mae_sum += (pred - true).abs().sum().item()
    return mae_sum / len(loader.dataset)              # eV


def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 epochs=EPOCHS, patience=PATIENCE, save_path=None, label='Model',
                 train_fn=None, writer=None, log_sparsity=False):
    \"\"\"
    Generic training loop with:
      - ReduceLROnPlateau scheduling
      - Early stopping
      - Best-checkpoint saving
      - Optional TensorBoard logging (pass a SummaryWriter as `writer`)
    Returns (history_dict, best_val_mae_in_eV).
    train_fn: optional override for train_epoch (used by RigL variant)
    log_sparsity: if True, also logs weight density to TensorBoard (for RigL)
    \"\"\"
    if train_fn is None:
        train_fn = train_epoch

    best_val = float('inf')
    wait     = 0
    history  = {'train_loss': [], 'val_mae_mev': []}

    for epoch in range(1, epochs + 1):
        train_loss = train_fn(model, train_loader, optimizer, device=DEVICE)
        val_mae    = evaluate(model, val_loader, device=DEVICE)          # eV
        lr_now     = optimizer.param_groups[0]['lr']

        if scheduler is not None:
            scheduler.step(val_mae)

        history['train_loss'].append(train_loss)
        history['val_mae_mev'].append(val_mae * 1000)

        # ── TensorBoard logging ───────────────────────────────────────────
        if writer is not None:
            writer.add_scalar('Loss/train',   train_loss,      epoch)
            writer.add_scalar('MAE/val_meV',  val_mae * 1000,  epoch)
            writer.add_scalar('LR',           lr_now,          epoch)
            if log_sparsity:
                tot = sum(p.numel() for p in model.parameters() if p.dim() > 1)
                nz  = sum((p != 0).sum().item() for p in model.parameters() if p.dim() > 1)
                writer.add_scalar('Sparsity', 1.0 - nz / max(tot, 1), epoch)

        if val_mae < best_val:
            best_val = val_mae
            wait = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{label}] Early stopping at epoch {epoch}.")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{label}] Ep {epoch:03d} | loss {train_loss:.5f} "
                  f"| val MAE {val_mae*1000:.2f} meV | lr {lr_now:.2e}")

    return history, best_val

print("Utilities defined.")"""
))

# ===========================================================================
# 6. TRAIN BASELINE
# ===========================================================================
cells.append(md("## 6 · Train Baseline GCN"))

cells.append(code(
"""model_baseline = GCN().to(DEVICE)
opt_b = torch.optim.Adam(model_baseline.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sch_b = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_b, patience=LR_PATIENCE, factor=0.5, min_lr=1e-5)

BASELINE_CKPT = os.path.join(CKPT_DIR, 'baseline_best.pt')
writer_b = SummaryWriter(os.path.join(LOG_DIR, 'baseline'))

print("=" * 65)
print("Baseline GCN")
print("=" * 65)

hist_baseline, best_val_baseline = run_training(
    model_baseline, train_loader, val_loader,
    opt_b, sch_b,
    epochs=EPOCHS, patience=PATIENCE,
    save_path=BASELINE_CKPT,
    label='Baseline',
    writer=writer_b,
)"""
))

cells.append(code(
"""# Load best checkpoint → evaluate on held-out test set.
model_baseline.load_state_dict(torch.load(BASELINE_CKPT))
test_mae_baseline = evaluate(model_baseline, test_loader, DEVICE)
print(f"\\nBaseline GCN  — Test MAE: {test_mae_baseline*1000:.2f} meV")

writer_b.add_scalar('MAE/test_meV', test_mae_baseline * 1000, 0)
writer_b.close()

# Save history so the reproduce cell can regenerate plots without retraining.
with open(os.path.join(SAVE_DIR, 'hist_baseline.json'), 'w') as f:
    json.dump(hist_baseline, f)
print("History saved → hist_baseline.json")"""
))

# ===========================================================================
# 7. VARIANT 1 — DENDRITIC GCN (PERFORATED AI)
# ===========================================================================
cells.append(md(
"""## 7 · Variant 1: Dendritic GCN (Perforated AI)

Beniaguev et al., *"Dendrites endow artificial neural networks with accurate, robust and
parameter-efficient learning"*, **Nature Communications 2025**.

**Hypothesis:** biological neurons perform rich local computation in their dendritic trees
before signals reach the soma. Perforated AI adds learnable dendritic branches to each
neuron, allowing the network to represent more complex input–output mappings with the same
number of soma-level parameters. For molecular property prediction, where subtle structural
motifs drive reactivity, this richer per-neuron computation may improve accuracy.

Perforated AI wraps any existing PyTorch model — only a few extra lines are needed.

> ⚠️ **API note:** if any `PBG`/`PBU` call raises an `AttributeError`, check the latest
> README at https://github.com/PerforatedAI/PerforatedAI and update accordingly."""
))

cells.append(code(
"""from perforatedai import pb_globals as PBG
from perforatedai import pb_utils  as PBU

# ── PAI global configuration ─────────────────────────────────────────────
PBG.pbTracker.setOptimizer(torch.optim.Adam)
PBG.pbTracker.setLearningRate(LR)

# ── Build GCN and wrap with Perforated AI dendrites ──────────────────────
model_dendritic = GCN().to(DEVICE)
model_dendritic = PBU.initializePB(model_dendritic)

# PAI manages the optimizer internally to coordinate dendritic updates.
optimizer_d, scheduler_d = PBG.pbTracker.setupOptimizer(
    model_dendritic,
    optimizerArgs={'lr': LR, 'weight_decay': WEIGHT_DECAY},
    schedulerArgs=None,
)

DENDRITIC_CKPT  = os.path.join(CKPT_DIR, 'dendritic_best.pt')
best_val_d      = float('inf')
hist_dendritic  = {'train_loss': [], 'val_mae_mev': []}
training_complete = False
epoch_d = 0
writer_d = SummaryWriter(os.path.join(LOG_DIR, 'dendritic'))

print("=" * 65)
print("Variant 1 — Dendritic GCN (Perforated AI)")
print("=" * 65)

while not training_complete and epoch_d < EPOCHS:
    epoch_d += 1

    # Standard forward/backward pass — no changes needed inside train_epoch.
    train_loss = train_epoch(model_dendritic, train_loader, optimizer_d, DEVICE)
    val_mae    = evaluate(model_dendritic, val_loader, DEVICE)            # eV

    hist_dendritic['train_loss'].append(train_loss)
    hist_dendritic['val_mae_mev'].append(val_mae * 1000)

    # TensorBoard logging
    writer_d.add_scalar('Loss/train',  train_loss,      epoch_d)
    writer_d.add_scalar('MAE/val_meV', val_mae * 1000,  epoch_d)
    writer_d.add_scalar('LR', optimizer_d.param_groups[0]['lr'], epoch_d)

    if val_mae < best_val_d:
        best_val_d = val_mae
        torch.save(model_dendritic.state_dict(), DENDRITIC_CKPT)

    # Report validation MAE to PAI tracker.
    # PAI decides when to grow dendrites and restructure the model.
    model_dendritic, restructured, training_complete = PBG.pbTracker.addToBestLoss(
        model_dendritic, val_mae
    )

    if restructured:
        # After adding dendritic branches, reinitialise the optimizer.
        optimizer_d, scheduler_d = PBG.pbTracker.setupOptimizer(
            model_dendritic,
            optimizerArgs={'lr': LR, 'weight_decay': WEIGHT_DECAY},
            schedulerArgs=None,
        )
        writer_d.add_scalar('Events/restructure', 1, epoch_d)
        print(f"  → Dendrites added / model restructured at epoch {epoch_d}")

    if epoch_d % 5 == 0 or epoch_d == 1:
        print(f"[Dendritic] Ep {epoch_d:03d} | loss {train_loss:.5f} "
              f"| val MAE {val_mae*1000:.2f} meV")"""
))

cells.append(code(
"""model_dendritic.load_state_dict(torch.load(DENDRITIC_CKPT))
test_mae_dendritic = evaluate(model_dendritic, test_loader, DEVICE)
print(f"\\nDendritic GCN — Test MAE: {test_mae_dendritic*1000:.2f} meV")

writer_d.add_scalar('MAE/test_meV', test_mae_dendritic * 1000, 0)
writer_d.close()

with open(os.path.join(SAVE_DIR, 'hist_dendritic.json'), 'w') as f:
    json.dump(hist_dendritic, f)
print("History saved → hist_dendritic.json")"""
))

# ===========================================================================
# 8. VARIANT 2 — SPARSE GCN (RIGL)
# ===========================================================================
cells.append(md(
"""## 8 · Variant 2: Sparse GCN (RigL)

Evci et al., *"Rigging the Lottery: Making All Tickets Winners"*, **ICML 2020**.

**Hypothesis:** many GCN weights trained on QM9 may be redundant — the molecules are small
(≤ 29 heavy atoms) and the relevant structural motifs are sparse. RigL enforces sparsity
*during* training: it periodically removes low-magnitude weights and regrows connections
where the gradient signal is strongest. Unlike post-hoc pruning, the topology evolves
throughout training, letting the model discover which connections matter most. A 50 % sparse
network also uses half the memory and fewer FLOPs at inference — relevant for large-scale
molecular screening.

Integration requires a one-line scheduler wrapping the optimizer; the training loop changes
by one conditional."""
))

cells.append(code(
"""from rigl_torch.RigL import RigLScheduler

# ── Build a fresh GCN (same architecture as baseline) ────────────────────
model_sparse = GCN().to(DEVICE)
opt_s        = torch.optim.Adam(model_sparse.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# T_end: stop updating topology at 75 % of total training steps.
T_end = int(0.75 * len(train_loader) * EPOCHS)

pruner = RigLScheduler(
    model_sparse,
    opt_s,
    dense_allocation      = 1.0 - RIGL_SPARSITY,   # fraction of weights to KEEP
    sparsity_distribution = 'uniform',
    T_end                 = T_end,
    delta                 = RIGL_DELTA,             # steps between topology updates
    alpha                 = RIGL_ALPHA,             # fraction of weights to redistribute
    static_topo           = False,                  # dynamic (not fixed) topology
    grad_accumulation_n   = 1,
    ignore_linear_layers  = False,
    keep_first_layer_dense= True,                   # first layer sees all features
)

sch_s = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_s, patience=LR_PATIENCE,
                                                    factor=0.5, min_lr=1e-5)

def train_epoch_rigl(model, loader, optimizer, device,
                     mean=None, std=None, target=TARGET_IDX):
    \"\"\"
    RigL training epoch.
    The only change vs. train_epoch: `pruner()` decides whether to step the
    optimizer (weights update) or update the sparsity mask — never both.
    \"\"\"
    mean = mean if mean is not None else MEAN
    std  = std  if std  is not None else STD
    model.train()
    total_loss = 0.0
    for data in loader:
        data   = data.to(device)
        y_norm = (data.y[:, target] - mean) / std
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out, y_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if pruner():            # True → weight step; False → mask update
            optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

SPARSE_CKPT = os.path.join(CKPT_DIR, 'sparse_best.pt')
writer_s = SummaryWriter(os.path.join(LOG_DIR, 'sparse'))

print("=" * 65)
print(f"Variant 2 — Sparse GCN (RigL, {int(RIGL_SPARSITY*100)}% sparsity)")
print("=" * 65)

hist_sparse, best_val_sparse = run_training(
    model_sparse, train_loader, val_loader,
    opt_s, sch_s,
    epochs=EPOCHS, patience=PATIENCE,
    save_path=SPARSE_CKPT,
    label='Sparse',
    train_fn=train_epoch_rigl,
    writer=writer_s,
    log_sparsity=True,
)"""
))

cells.append(code(
"""model_sparse.load_state_dict(torch.load(SPARSE_CKPT))
test_mae_sparse = evaluate(model_sparse, test_loader, DEVICE)
print(f"\\nSparse GCN (RigL) — Test MAE: {test_mae_sparse*1000:.2f} meV")

writer_s.add_scalar('MAE/test_meV', test_mae_sparse * 1000, 0)
writer_s.close()

with open(os.path.join(SAVE_DIR, 'hist_sparse.json'), 'w') as f:
    json.dump(hist_sparse, f)
print("History saved → hist_sparse.json")"""
))

# ---------------------------------------------------------------------------
# TensorBoard viewer  (launch after training to inspect all runs)
# ---------------------------------------------------------------------------
cells.append(md(
"""## 8b · TensorBoard — Inspect Training Runs

Run the cell below to launch an interactive TensorBoard dashboard inside Colab.
All three runs are logged to `LOG_DIR/runs/` and persist in your Google Drive."""
))

cells.append(code(
"""%tensorboard --logdir {LOG_DIR}"""
))

# ===========================================================================
# 9. RESULTS
# ===========================================================================
cells.append(md("## 9 · Results and Analysis"))

cells.append(code(
"""SOTA_MEV = 50.0   # approximate SotA reference (meV) for HOMO-LUMO gap on QM9

results = {
    'Baseline GCN'       : test_mae_baseline   * 1000,
    'Dendritic GCN'      : test_mae_dendritic  * 1000,
    'Sparse GCN (RigL)'  : test_mae_sparse     * 1000,
}

print("\\n" + "=" * 60)
print(f"{'Model':<24} {'Test MAE (meV)':>14} {'vs Baseline':>13}")
print("-" * 60)
for name, mae in results.items():
    if name == 'Baseline GCN':
        delta_str = '—'
    else:
        delta = mae - results['Baseline GCN']
        delta_str = f"{delta:+.2f} meV"
    print(f"{name:<24} {mae:>14.2f} {delta_str:>13}")
print("-" * 60)
print(f"{'SotA (GNN reference)':<24} {SOTA_MEV:>14.1f} {'':>13}")
print("=" * 60)

# Persist results for reproduce_results script
with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
    json.dump({k: round(v, 3) for k, v in results.items()}, f, indent=2)
print("\\nResults saved to results.json")"""
))

# ---------------------------------------------------------------------------
# Learning curve plot
# ---------------------------------------------------------------------------
cells.append(code(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
COLORS = {'Baseline GCN': '#4C72B0', 'Dendritic GCN': '#DD8452', 'Sparse GCN (RigL)': '#55A868'}

# ── Left: Validation MAE curves ──────────────────────────────────────────
ax = axes[0]
for label, hist in [('Baseline GCN',       hist_baseline),
                    ('Dendritic GCN',       hist_dendritic),
                    ('Sparse GCN (RigL)',   hist_sparse)]:
    ax.plot(hist['val_mae_mev'], label=label, color=COLORS[label], linewidth=2)
ax.axhline(SOTA_MEV, color='gray', linestyle='--', linewidth=1.2, label='SotA ref. (50 meV)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation MAE (meV)', fontsize=12)
ax.set_title('Validation MAE vs. Epoch', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── Right: Test MAE bar chart ─────────────────────────────────────────────
ax = axes[1]
names  = list(results.keys())
maes   = list(results.values())
colors = [COLORS[n] for n in names]
bars   = ax.bar(names, maes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
ax.axhline(SOTA_MEV, color='gray', linestyle='--', linewidth=1.2, label='SotA ref.')
for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.4,
            f'{mae:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Test MAE (meV)', fontsize=12)
ax.set_title('Test MAE Comparison', fontsize=13)
ax.set_ylim(0, max(maes) * 1.35)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=10, ha='right', fontsize=10)

plt.tight_layout()
FIG_PATH = os.path.join(SAVE_DIR, 'results_comparison.pdf')
plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figure saved: {FIG_PATH}")"""
))

# ---------------------------------------------------------------------------
# Sparsity analysis
# ---------------------------------------------------------------------------
cells.append(md(
"""### Sparsity Analysis (RigL)

The table below shows layer-wise weight density in the trained sparse model.
This is useful for the report's Results & Discussion section — you can comment on
*which* layers RigL found most compressible and why."""
))

cells.append(code(
"""print(f"\\n{'Layer':<42} {'Non-zero':>10} {'Total':>10} {'Density':>9}")
print("-" * 75)
total_nz, total_params = 0, 0
for name, param in model_sparse.named_parameters():
    if param.dim() > 1:                       # skip bias vectors
        nz     = (param != 0).sum().item()
        total  = param.numel()
        density = nz / total
        total_nz     += nz
        total_params += total
        print(f"{name:<42} {nz:>10,} {total:>10,} {density:>8.1%}")
print("-" * 75)
overall = total_nz / total_params if total_params > 0 else 0
print(f"{'Overall':42} {total_nz:>10,} {total_params:>10,} {overall:>8.1%}")"""
))

# ---------------------------------------------------------------------------
# Parameter count comparison
# ---------------------------------------------------------------------------
cells.append(code(
"""# Count trainable parameters for each model (for the Methods table in your report).
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\\nParameter counts:")
print(f"  Baseline GCN    : {count_params(model_baseline):>10,}")
print(f"  Dendritic GCN   : {count_params(model_dendritic):>10,}  (includes dendritic branches)")
print(f"  Sparse GCN      : {count_params(model_sparse):>10,}  (same arch; ~{int(RIGL_SPARSITY*100)}% zeroed)")"""
))

# ===========================================================================
# 10. REPRODUCE RESULTS
# ===========================================================================
cells.append(md(
"""## 10 · Reproduce Results  *(standalone — run without retraining)*

**This section satisfies the `reproduce_results` submission requirement (§5.3).**

It is fully self-contained: given only the saved checkpoints and history files it
regenerates all reported metrics *and* all figures without executing any training cells.

**How to use after a session restart:**
1. Run cells 1–5 (setup, installs, imports, config, data).
2. Jump straight to this section and run the two cells below."""
))

_repro_eval_src = "\n".join([
    "# 10a. Load checkpoints and re-evaluate on test set",
    "# Requires: checkpoints saved to CKPT_DIR during training.",
    "# Run cells 1-5 first (setup, installs, imports, config, data), then run this.",
    "",
    'print("=" * 65)',
    'print("Reproducing results from saved checkpoints")',
    'print("=" * 65)',
    "",
    "r_baseline  = GCN().to(DEVICE)",
    "r_dendritic = GCN().to(DEVICE)",
    "r_sparse    = GCN().to(DEVICE)",
    "",
    "r_baseline.load_state_dict(",
    '    torch.load(os.path.join(CKPT_DIR, "baseline_best.pt"),  map_location=DEVICE))',
    "r_dendritic.load_state_dict(",
    '    torch.load(os.path.join(CKPT_DIR, "dendritic_best.pt"), map_location=DEVICE))',
    "r_sparse.load_state_dict(",
    '    torch.load(os.path.join(CKPT_DIR, "sparse_best.pt"),    map_location=DEVICE))',
    "",
    "repro_results = {",
    '    "Baseline GCN"      : evaluate(r_baseline,  test_loader, DEVICE) * 1000,',
    '    "Dendritic GCN"     : evaluate(r_dendritic, test_loader, DEVICE) * 1000,',
    '    "Sparse GCN (RigL)" : evaluate(r_sparse,    test_loader, DEVICE) * 1000,',
    "}",
    "",
    "SOTA_MEV = 50.0",
    'header = "{:<24} {:>14} {:>13}".format("Model", "Test MAE (meV)", "vs Baseline")',
    'print("\\n" + header)',
    'print("-" * 55)',
    'baseline_mae = repro_results["Baseline GCN"]',
    "for name, mae in repro_results.items():",
    '    if name == "Baseline GCN":',
    '        delta_str = "    -"',
    "    else:",
    '        delta_str = "{:+.2f} meV".format(mae - baseline_mae)',
    '    print("{:<24} {:>14.2f} {:>13}".format(name, mae, delta_str))',
    'print("-" * 55)',
    'print("{:<24} {:>14.1f}".format("SotA (GNN reference)", SOTA_MEV))',
])
cells.append(code(_repro_eval_src))

_repro_fig_src = "\n".join([
    "# 10b. Regenerate all figures from saved history files",
    "# Requires: hist_baseline.json, hist_dendritic.json, hist_sparse.json",
    "# (written automatically after each model's training cell).",
    "",
    "with open(os.path.join(SAVE_DIR, 'hist_baseline.json'))  as f: rh_b = json.load(f)",
    "with open(os.path.join(SAVE_DIR, 'hist_dendritic.json')) as f: rh_d = json.load(f)",
    "with open(os.path.join(SAVE_DIR, 'hist_sparse.json'))    as f: rh_s = json.load(f)",
    "",
    "COLORS = {'Baseline GCN': '#4C72B0', 'Dendritic GCN': '#DD8452', 'Sparse GCN (RigL)': '#55A868'}",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
    "",
    "# Left: Validation MAE learning curves",
    "ax = axes[0]",
    "for label, hist in [('Baseline GCN', rh_b), ('Dendritic GCN', rh_d), ('Sparse GCN (RigL)', rh_s)]:",
    "    ax.plot(hist['val_mae_mev'], label=label, color=COLORS[label], linewidth=2)",
    "ax.axhline(SOTA_MEV, color='gray', linestyle='--', linewidth=1.2, label='SotA ref. (50 meV)')",
    "ax.set_xlabel('Epoch', fontsize=12)",
    "ax.set_ylabel('Validation MAE (meV)', fontsize=12)",
    "ax.set_title('Validation MAE vs. Epoch', fontsize=13)",
    "ax.legend(fontsize=10)",
    "ax.grid(True, alpha=0.3)",
    "",
    "# Right: Test MAE bar chart",
    "ax = axes[1]",
    "names  = list(repro_results.keys())",
    "maes   = list(repro_results.values())",
    "colors = [COLORS[n] for n in names]",
    "bars   = ax.bar(names, maes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)",
    "ax.axhline(SOTA_MEV, color='gray', linestyle='--', linewidth=1.2, label='SotA ref.')",
    "for bar, mae in zip(bars, maes):",
    "    label_txt = '{:.1f}'.format(mae)",
    "    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.4,",
    "            label_txt, ha='center', va='bottom', fontweight='bold', fontsize=11)",
    "ax.set_ylabel('Test MAE (meV)', fontsize=12)",
    "ax.set_title('Test MAE Comparison', fontsize=13)",
    "ax.set_ylim(0, max(maes) * 1.35)",
    "ax.legend(fontsize=10)",
    "ax.grid(True, alpha=0.3, axis='y')",
    "ax.set_xticks(range(len(names)))",
    "ax.set_xticklabels(names, rotation=10, ha='right', fontsize=10)",
    "",
    "plt.tight_layout()",
    "FIG_PATH = os.path.join(SAVE_DIR, 'results_comparison.pdf')",
    "plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')",
    "plt.show()",
    'print("Figure saved: " + FIG_PATH)',
])
cells.append(code(_repro_fig_src))

# ===========================================================================
# Build notebook JSON
# ===========================================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "collapsed_sections": []
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

# Ensure source fields are lists of strings (required by nbformat)
for cell in nb["cells"]:
    src = cell["source"]
    if isinstance(src, str):
        cell["source"] = [line + "\n" for line in src.split("\n")]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {OUT}")
