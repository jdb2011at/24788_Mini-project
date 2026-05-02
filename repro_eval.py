# 10a. Load checkpoints and re-evaluate on test set
# Requires: checkpoints saved to CKPT_DIR during training.
# Run cells 1-5 first (setup, installs, imports, config, data), then this cell.

print("=" * 65)
print("Reproducing results from saved checkpoints")
print("=" * 65)

r_baseline  = GCN().to(DEVICE)
r_dendritic = GCN().to(DEVICE)
r_sparse    = GCN().to(DEVICE)

r_baseline.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, "baseline_best.pt"),
               map_location=DEVICE))
r_dendritic.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, "dendritic_best.pt"),
               map_location=DEVICE))
r_sparse.load_state_dict(
    torch.load(os.path.join(CKPT_DIR, "sparse_best.pt"),
               map_location=DEVICE))

repro_results = {
    "Baseline GCN"      : evaluate(r_baseline,  test_loader, DEVICE) * 1000,
    "Dendritic GCN"     : evaluate(r_dendritic, test_loader, DEVICE) * 1000,
    "Sparse GCN (RigL)" : evaluate(r_sparse,    test_loader, DEVICE) * 1000,
}

SOTA_MEV = 50.0
header = "{:<24} {:>14} {:>13}".format(
    "Model", "Test MAE (meV)", "vs Baseline")
print("\n" + header)
print("-" * 55)
baseline_mae = repro_results["Baseline GCN"]
for name, mae in repro_results.items():
    if name == "Baseline GCN":
        delta_str = "    -"
    else:
        delta_str = "{:+.2f} meV".format(mae - baseline_mae)
    print("{:<24} {:>14.2f} {:>13}".format(name, mae, delta_str))
print("-" * 55)
print("{:<24} {:>14.1f}".format("SotA (GNN reference)", SOTA_MEV))
