# train_split_dynamic.py
"""
TASK 3 (Approach 2B): Split/Branched Output-Gated ANN - Dynamic Hyperparameter Tuning
Exploits the decoupled nature of the Output-Gated Branched Network to dynamically
tune the architecture, activation, L2 regularisation (alpha), and learning rate
INDEPENDENTLY for the SCARA branch and the Ned branch on their validation sets,
then packages them into a single unified SplitBranchedIKNetwork.
Saves: models/model_split_dynamic_{small,medium,large}.pkl
       models/search_split_dynamic.csv
"""

import os
import csv
import json
import pickle
import itertools
from train_single_fixed import load_combined_dataset, MODEL_DIR
from train_split_fixed import train_single_branch, SplitBranchedIKNetwork

CANDIDATE_BRANCH_LAYERS = [
    (64, 32),
    (128, 64),
    (128, 64, 32),
    (128, 128, 64),
]
CANDIDATE_ACTIVATIONS = ["tanh", "relu"]
CANDIDATE_ALPHAS = [1e-3, 1e-4, 1e-5]
CANDIDATE_LR_MAX = [1e-3, 3e-3]

SCREEN_EPOCHS = 200
FINAL_EPOCHS = 350


def select_best_branch_topology(X_tr_m, y_tr_m, X_val_m, y_val_m, arm_label, search_rows):
    """Dynamically finds the best (hidden_layer_sizes, activation) for one robot branch."""
    print(f"\n--- Stage 1: Dynamic Topology Search for [{arm_label.upper()} Branch] (Medium Val Set) ---")
    print(f"{'Branch Layers':<16} | {'Act':<5} | {'Best Ep':<7} | {'Branch Val MSE':<14}")
    print("-" * 52)
    best_cfg = None
    best_mse = float("inf")

    for layers, act in itertools.product(CANDIDATE_BRANCH_LAYERS, CANDIDATE_ACTIVATIONS):
        _, _, hist = train_single_branch(
            X_tr_m, y_tr_m, X_val_m, y_val_m,
            hidden_layer_sizes=layers,
            activation=act,
            alpha=1e-4,
            batch_size=64,
            epochs=SCREEN_EPOCHS,
            lr_max=3e-3,
        )
        ep = hist["best_epoch"]
        v_mse = hist["best_val_mse"]
        print(f"{str(layers):<16} | {act:<5} | {ep:<7d} | {v_mse:<14.6f}")

        search_rows.append({
            "stage": "branch_topology_search", "arm": arm_label, "size": "medium",
            "layers": str(layers), "activation": act,
            "alpha": 1e-4, "lr_max": 3e-3, "best_epoch": ep, "val_mse": v_mse,
        })

        if v_mse < best_mse:
            best_mse = v_mse
            best_cfg = {"hidden_layer_sizes": layers, "activation": act}

    print(f"--> Selected {arm_label.upper()} Branch Topology: {best_cfg['hidden_layer_sizes']} + '{best_cfg['activation']}' (MSE: {best_mse:.6f})")
    return best_cfg


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 85)
    print("APPROACH 2B: Split/Branched ANN - Per-Branch Dynamic Hyperparameter Search")
    print("=" * 85)

    search_rows = []

    # Stage 1: Select best branch topology independently for SCARA and Ned on Medium Set
    X_tr_m, y_tr_m, X_val_m, y_val_m = load_combined_dataset("medium")
    tr_s, tr_n = X_tr_m[:, 0] > 0.5, X_tr_m[:, 1] > 0.5
    val_s, val_n = X_val_m[:, 0] > 0.5, X_val_m[:, 1] > 0.5

    best_scara_arch = select_best_branch_topology(
        X_tr_m[tr_s, 2:5], y_tr_m[tr_s], X_val_m[val_s, 2:5], y_val_m[val_s], "scara", search_rows
    )
    best_ned_arch = select_best_branch_topology(
        X_tr_m[tr_n, 2:5], y_tr_m[tr_n], X_val_m[val_n, 2:5], y_val_m[val_n], "ned", search_rows
    )

    # Stage 2: Dynamically tune (alpha, lr_max) per branch across Small, Medium, Large
    batch_map = {"small": 32, "medium": 64, "large": 128}
    histories = {}

    for size_name in ["small", "medium", "large"]:
        X_tr, y_tr, X_val, y_val = load_combined_dataset(size_name)
        tr_s, tr_n = X_tr[:, 0] > 0.5, X_tr[:, 1] > 0.5
        val_s, val_n = X_val[:, 0] > 0.5, X_val[:, 1] > 0.5
        b_size = batch_map[size_name]

        print(f"\n>>> Stage 2: Dynamic Branch Tuning for [{size_name.upper()}] Dataset <<<")

        best_branch_models = {}
        for arm_label, arch, X_tr_sub, y_tr_sub, X_val_sub, y_val_sub, seed_off in [
            ("scara", best_scara_arch, X_tr[tr_s, 2:5], y_tr[tr_s], X_val[val_s, 2:5], y_val[val_s], 0),
            ("ned",   best_ned_arch,   X_tr[tr_n, 2:5], y_tr[tr_n], X_val[val_n, 2:5], y_val[val_n], 100),
        ]:
            best_mse = float("inf")
            best_tuple = None
            best_hp = None

            for alpha, lr_max in itertools.product(CANDIDATE_ALPHAS, CANDIDATE_LR_MAX):
                scaler, mlp, hist = train_single_branch(
                    X_tr_sub, y_tr_sub, X_val_sub, y_val_sub,
                    hidden_layer_sizes=arch["hidden_layer_sizes"],
                    activation=arch["activation"],
                    alpha=alpha,
                    batch_size=b_size,
                    epochs=FINAL_EPOCHS,
                    lr_max=lr_max,
                    random_state=42 + seed_off,
                )
                ep = hist["best_epoch"]
                v_mse = hist["best_val_mse"]

                search_rows.append({
                    "stage": "size_tuning", "arm": arm_label, "size": size_name,
                    "layers": str(arch["hidden_layer_sizes"]), "activation": arch["activation"],
                    "alpha": alpha, "lr_max": lr_max, "best_epoch": ep, "val_mse": v_mse,
                })

                if v_mse < best_mse:
                    best_mse = v_mse
                    best_tuple = (scaler, mlp, hist)
                    best_hp = {**arch, "alpha": alpha, "lr_max": lr_max, "batch_size": b_size, "best_epoch": ep}

            best_branch_models[arm_label] = (best_tuple, best_hp)
            print(
                f"  [{arm_label.upper():<5} Winner] Layers={best_hp['hidden_layer_sizes']}, "
                f"alpha={best_hp['alpha']:.0e}, lr={best_hp['lr_max']}, "
                f"ep={best_hp['best_epoch']} -> Val MSE: {best_mse:.6f}"
            )

        # Assemble winning branches into a single unified SplitBranchedIKNetwork
        (s_scaler, s_mlp, s_hist), s_hp = best_branch_models["scara"]
        (n_scaler, n_mlp, n_hist), n_hp = best_branch_models["ned"]

        unified_model = SplitBranchedIKNetwork(scara_params=s_hp, ned_params=n_hp)
        unified_model.scara_scaler = s_scaler
        unified_model.scara_mlp = s_mlp
        unified_model.ned_scaler = n_scaler
        unified_model.ned_mlp = n_mlp
        unified_model.history = {
            "scara": s_hist,
            "ned": n_hist,
            "best_val_mse": 0.5 * (s_hist["best_val_mse"] + n_hist["best_val_mse"]),
            "selected_scara_hp": s_hp,
            "selected_ned_hp": n_hp,
        }

        print(f"  [COMBINED {size_name.upper()}] Combined Val MSE: {unified_model.history['best_val_mse']:.6f}")

        out_path = os.path.join(MODEL_DIR, f"model_split_dynamic_{size_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(unified_model, f)
        histories[size_name] = unified_model.history

    csv_path = os.path.join(MODEL_DIR, "search_split_dynamic.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    with open(os.path.join(MODEL_DIR, "history_split_dynamic.json"), "w") as f:
        json.dump(histories, f, indent=2)
    print(f"\nSaved dynamic branch search table -> {csv_path}")


if __name__ == "__main__":
    main()