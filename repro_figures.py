# 10b. Regenerate all figures from saved history files
# Requires: hist_baseline.json, hist_dendritic.json, hist_sparse.json
# (written automatically after each model's training cell).

with open(os.path.join(SAVE_DIR, "hist_baseline.json"))  as f:
    rh_b = json.load(f)
with open(os.path.join(SAVE_DIR, "hist_dendritic.json")) as f:
    rh_d = json.load(f)
with open(os.path.join(SAVE_DIR, "hist_sparse.json"))    as f:
    rh_s = json.load(f)

COLORS = {
    "Baseline GCN"      : "#4C72B0",
    "Dendritic GCN"     : "#DD8452",
    "Sparse GCN (RigL)" : "#55A868",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Validation MAE learning curves
ax = axes[0]
pairs = [
    ("Baseline GCN",      rh_b),
    ("Dendritic GCN",     rh_d),
    ("Sparse GCN (RigL)", rh_s),
]
for label, hist in pairs:
    ax.plot(hist["val_mae_mev"], label=label,
            color=COLORS[label], linewidth=2)
ax.axhline(SOTA_MEV, color="gray", linestyle="--",
           linewidth=1.2, label="SotA ref. (50 meV)")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation MAE (meV)", fontsize=12)
ax.set_title("Validation MAE vs. Epoch", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: Test MAE bar chart
ax = axes[1]
names  = list(repro_results.keys())
maes   = list(repro_results.values())
colors = [COLORS[n] for n in names]
bars   = ax.bar(names, maes, color=colors,
                alpha=0.85, edgecolor="black", linewidth=0.8)
ax.axhline(SOTA_MEV, color="gray", linestyle="--",
           linewidth=1.2, label="SotA ref.")
for bar, mae in zip(bars, maes):
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        bar.get_height() + 0.4,
        "{:.1f}".format(mae),
        ha="center", va="bottom", fontweight="bold", fontsize=11,
    )
ax.set_ylabel("Test MAE (meV)", fontsize=12)
ax.set_title("Test MAE Comparison", fontsize=13)
ax.set_ylim(0, max(maes) * 1.35)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=10, ha="right", fontsize=10)

plt.tight_layout()
FIG_PATH = os.path.join(SAVE_DIR, "results_comparison.pdf")
plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved: " + FIG_PATH)
