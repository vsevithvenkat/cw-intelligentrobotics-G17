# evaluate_4b_ned.py
"""
TASK 4.B: Webots Simulation Evaluation for Niryo Ned (First 100 Test Samples)
Creates the 100-sample reduced test set (datasets/ned_test_100.csv), predicts
joint values using each trained model (small, medium, large), commands the
simulated Ned arm in Webots, reads the Supervisor TCP position, and reports
Mean ± Std position error (comparing 4.B Webots error vs 4.A FK error).
"""

import os
import sys
import csv
import pickle
import numpy as np

# Ensure root project directory is on sys.path if running from controllers/<name>/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.exists(os.path.join(PROJECT_ROOT, "datasets")):
    ROOT_DIR = PROJECT_ROOT
else:
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ned_controller import NedController
from ned_fk import ned_forward_kinematics
from generate_datasets import get_norm_limits, denormalise_joints
from train_single_fixed import SingleSharedIKNetwork
from train_split_fixed import SplitBranchedIKNetwork

DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

APPROACHES = ["single_fixed", "single_dynamic", "split_fixed", "split_dynamic"]
SIZES = ["small", "medium", "large"]
N_TEST_4B = 100


def create_and_load_100_test_set(arm_name="ned"):
    """
    Task 4.B Step 1: Create new test set using only the first 100 samples
    from the original test set.
    """
    orig_path = os.path.join(DATASET_DIR, f"{arm_name}_test.csv")
    sub_path = os.path.join(DATASET_DIR, f"{arm_name}_test_100.csv")

    with open(orig_path, "r", newline="") as f_in:
        reader = list(csv.reader(f_in))
        header = reader[0]
        rows_100 = reader[1 : 1 + N_TEST_4B]

    with open(sub_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(rows_100)

    data = np.array(rows_100, dtype=float)
    X_test_100 = data[:, 0:5]    # [is_scara, is_ned, x, y, z]
    xyz_true_100 = data[:, 2:5]  # [x, y, z]_true
    return X_test_100, xyz_true_100, sub_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    controller = NedController()
    limits = get_norm_limits("ned")

    X_test, xyz_true, sub_path = create_and_load_100_test_set("ned")
    print("=" * 100)
    print(f"TASK 4.B (NIRYO NED): Webots Simulation Evaluation on First {len(X_test)} Test Samples")
    print(f"Saved 100-sample test set -> {sub_path}")
    print("=" * 100)

    summary_rows = []

    for approach in APPROACHES:
        first_path = os.path.join(MODEL_DIR, f"model_{approach}_small.pkl")
        if not os.path.exists(first_path):
            if approach == "single_fixed" and os.path.exists(os.path.join(MODEL_DIR, "model_small.pkl")):
                prefix = "model"
            else:
                continue
        else:
            prefix = f"model_{approach}"

        print(f"\n>>> Evaluating Approach: {approach.upper()} on NIRYO NED <<<")
        print(f"{'Model Size':<12} | {'4.B Webots Error (Mean ± Std mm)':<34} | {'4.A FK Error on Same 100 (mm)':<32} | {'Sim-FK Diff (mm)'}")
        print("-" * 100)

        for size_name in SIZES:
            model_path = os.path.join(MODEL_DIR, f"{prefix}_{size_name}.pkl")
            if not os.path.exists(model_path):
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Predict normalised joints and denormalise to radians
            q_norm_pred = model.predict(X_test)
            q_raw_pred = denormalise_joints(q_norm_pred, limits)

            sim_errors_m = []
            fk_errors_m = []
            sim_fk_diffs_m = []

            for i in range(len(X_test)):
                q_cmd = q_raw_pred[i]
                target_xyz = xyz_true[i]

                # Command Webots controller and read resulting Supervisor TCP
                sim_tcp = controller.move_to_joints_and_settle(q_cmd, max_steps=600)
                fk_tcp = ned_forward_kinematics(q_cmd)

                sim_errors_m.append(np.linalg.norm(target_xyz - sim_tcp))
                fk_errors_m.append(np.linalg.norm(target_xyz - fk_tcp))
                sim_fk_diffs_m.append(np.linalg.norm(sim_tcp - fk_tcp))

            sim_errors_m = np.array(sim_errors_m)
            fk_errors_m = np.array(fk_errors_m)
            sim_fk_diffs_m = np.array(sim_fk_diffs_m)

            sim_mean_mm = float(np.mean(sim_errors_m) * 1000.0)
            sim_std_mm = float(np.std(sim_errors_m) * 1000.0)
            fk_mean_mm = float(np.mean(fk_errors_m) * 1000.0)
            fk_std_mm = float(np.std(fk_errors_m) * 1000.0)
            diff_mean_mm = float(np.mean(sim_fk_diffs_m) * 1000.0)

            print(
                f"{size_name:<12} | {sim_mean_mm:8.3f} ± {sim_std_mm:8.3f} mm             | "
                f"{fk_mean_mm:8.3f} ± {fk_std_mm:8.3f} mm           | {diff_mean_mm:8.4f} mm"
            )

            summary_rows.append({
                "approach": approach,
                "model_size": size_name,
                "arm": "ned",
                "webots_pos_mean_m": float(np.mean(sim_errors_m)),
                "webots_pos_std_m": float(np.std(sim_errors_m)),
                "webots_pos_mean_mm": sim_mean_mm,
                "webots_pos_std_mm": sim_std_mm,
                "fk_100_pos_mean_mm": fk_mean_mm,
                "fk_100_pos_std_mm": fk_std_mm,
                "sim_vs_fk_discrepancy_mm": diff_mean_mm,
            })

    out_csv = os.path.join(RESULTS_DIR, "task_4b_ned_summary.csv")
    if summary_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nSaved Ned Task 4.B results -> {out_csv}")


if __name__ == "__main__":
    main()