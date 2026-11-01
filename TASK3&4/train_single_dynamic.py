# train_single_dynamic.py
"""
TASK 3 (Approach 1B): Single Shared Monolithic ANN - Dynamic Hyperparameter Tuning
Dynamically searches across hidden layer sizes, activations, L2 regularisation (alpha),
and learning rates on the Task 2 validation sets to select the best Single ANN.
Saves: models/model_single_dynamic_{small,medium,large}.pkl
       models/search_single_dynamic.csv
"""

import os
import csv
import json
import pickle
import itertools
from train_single_fixed import (
    load_combined_dataset,
    SingleSharedIKNetwork,
    MODEL_DIR,
)

# Candidate Hyperparameter Search Space
CANDIDATE_LAYERS = [
    (64, 32),
    (128, 64),
    (128, 64, 32),
    (256, 128, 64),
]
CANDIDATE_ACTIVATIONS = ["tanh", "relu"]
CANDIDATE_ALPHAS = [1e-3, 1e-4, 1e-5]
CANDIDATE_LR_MAX = [1e-3, 3e-3]

SCREEN_EPOCHS = 200
FINAL_EPOCHS = 350


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 85)
    print("APPROACH 1B: Single Shared ANN - Dynamic Hyperparameter Search")
    print("=" * 85)

    # Stage 1: Dynamically select the best architecture & activation on Medium Validation Set
    X_tr_m, y_tr_m, X_val_m, y_val_m = load_combined_dataset("medium")
    search_rows = []
    best_arch = None
    best_arch_mse = float("inf")

    print("\n--- Stage 1: Dynamic Topology & Activation Search (Medium Validation Set) ---")
    print(f"{'Hidden Layers':<16} | {'Act':<5} | {'Best Ep':<7} | {'Val MSE':<12} | {'SCARA MSE':<12} | {'Ned MSE':<12}")
    print("-" * 78)

    for layers, act in itertools.product(CANDIDATE_LAYERS, CANDIDATE_ACTIVATIONS):
        model = SingleSharedIKNetwork(
            hidden_layer_sizes=layers,
            activation=act,
            alpha=1e-4,
            batch_size=128,
            epochs=SCREEN_EPOCHS,
            lr_max=3e-3,
        ).fit(X_tr_m, y_tr_m, X_val_m, y_val_m, verbose=False)

        ep = model.history["best_epoch"]
        v_mse = model.history["best_val_mse"]
        s_mse = model.history["val_scara_mse"][ep - 1]
        n_mse = model.history["val_ned_mse"][ep - 1]

        print(f"{str(layers):<16} | {act:<5} | {ep:<7d} | {v_mse:<12.6f} | {s_mse:<12.6f} | {n_mse:<12.6f}")
        search_rows.append({
            "stage": "topology_search", "size": "medium",
            "layers": str(layers), "activation": act,
            "alpha": 1e-4, "lr_max": 3e-3, "best_epoch": ep,
            "val_mse": v_mse, "scara_mse": s_mse, "ned_mse": n_mse,
        })

        if v_mse < best_arch_mse:
            best_arch_mse = v_mse
            best_arch = {"layers": layers, "activation": act}

    print(f"--> Selected Optimal Topology: {best_arch['layers']} + '{best_arch['activation']}' (Val MSE: {best_arch_mse:.6f})")

    # Stage 2: Dynamically tune (alpha, lr_max, epochs) for Small, Medium, and Large
    batch_map = {"small": 64, "medium": 128, "large": 256}
    histories = {}

    for size_name in ["small", "medium", "large"]:
        X_tr, y_tr, X_val, y_val = load_combined_dataset(size_name)
        print(f"\n--- Stage 2: Dynamic Hyperparameter Tuning for [{size_name.upper()}] ---")
        print(f"{'Alpha':<10} | {'LR Max':<10} | {'Best Ep':<7} | {'Val MSE':<12} | {'SCARA MSE':<12} | {'Ned MSE':<12}")
        print("-" * 75)

        best_model = None
        best_mse = float("inf")
        best_cfg = None

        for alpha, lr_max in itertools.product(CANDIDATE_ALPHAS, CANDIDATE_LR_MAX):
            cand = SingleSharedIKNetwork(
                hidden_layer_sizes=best_arch["layers"],
                activation=best_arch["activation"],
                alpha=alpha,
                batch_size=batch_map[size_name],
                epochs=FINAL_EPOCHS,
                lr_max=lr_max,
            ).fit(X_tr, y_tr, X_val, y_val, verbose=False)

            ep = cand.history["best_epoch"]
            v_mse = cand.history["best_val_mse"]
            s_mse = cand.history["val_scara_mse"][ep - 1]
            n_mse = cand.history["val_ned_mse"][ep - 1]

            print(f"{alpha:<10.0e} | {lr_max:<10.4f} | {ep:<7d} | {v_mse:<12.6f} | {s_mse:<12.6f} | {n_mse:<12.6f}")
            search_rows.append({
                "stage": "size_tuning", "size": size_name,
                "layers": str(best_arch["layers"]), "activation": best_arch["activation"],
                "alpha": alpha, "lr_max": lr_max, "best_epoch": ep,
                "val_mse": v_mse, "scara_mse": s_mse, "ned_mse": n_mse,
            })

            if v_mse < best_mse:
                best_mse = v_mse
                best_model = cand
                best_cfg = {"alpha": alpha, "lr_max": lr_max, "best_epoch": ep, **best_arch}

        print(f"--> Winner [{size_name.upper()}]: alpha={best_cfg['alpha']:.0e}, lr={best_cfg['lr_max']}, ep={best_cfg['best_epoch']} (Val MSE={best_mse:.6f})")

        out_path = os.path.join(MODEL_DIR, f"model_single_dynamic_{size_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(best_model, f)
        histories[size_name] = {"selected_config": best_cfg, "history": best_model.history}

    # Save search logs
    csv_path = os.path.join(MODEL_DIR, "search_single_dynamic.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    with open(os.path.join(MODEL_DIR, "history_single_dynamic.json"), "w") as f:
        json.dump(histories, f, indent=2)
    print(f"\nSaved dynamic search table -> {csv_path}")


if __name__ == "__main__":
    main()