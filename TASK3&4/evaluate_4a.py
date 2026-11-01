# evaluate_4a.py
"""
TASK 4.A: Analytical Forward Kinematics Evaluation (No Simulator Required)
Evaluates trained models (small, medium, large) across all 4 training approaches
('single_fixed', 'single_dynamic', 'split_fixed', 'split_dynamic') on the
independent 500-sample test sets for Ned and SCARA.
Outputs:
  - 6 sets of position-error statistics per approach (3 models x 2 arms)
  - 18 sets of joint-error statistics per approach (3 models x 2 arms x 3 joints)
  - Saves results to results/task_4a_summary.csv and results/task_4a_summary.json
"""

import os
import csv
import json
import pickle
import numpy as np

from ned_fk import ned_forward_kinematics
from scara_fk import scara_forward_kinematics
from generate_datasets import get_norm_limits, denormalise_joints

# Import model classes so pickle.load can deserialize all 4 variants cleanly
from train_single_fixed import SingleSharedIKNetwork
from train_split_fixed import SplitBranchedIKNetwork

DATASET_DIR = "datasets"
MODEL_DIR = "models"
RESULTS_DIR = "results"

APPROACHES = ["single_fixed", "single_dynamic", "split_fixed", "split_dynamic"]
SIZES = ["small", "medium", "large"]
ARMS = ["ned", "scara"]


def load_test_set(arm_name, dataset_dir=DATASET_DIR):
    """
    Loads the test CSV for 'ned' or 'scara'.
    Columns: is_scara, is_ned, x, y, z, q1_norm, q2_norm, q3_norm, q1_raw, q2_raw, q3_raw
    """
    filepath = os.path.join(dataset_dir, f"{arm_name}_test.csv")
    data = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=float)
    X_test = data[:, 0:5]       # [is_scara, is_ned, x, y, z]
    xyz_true = data[:, 2:5]     # [x, y, z]_true
    q_norm_true = data[:, 5:8]  # [q1_norm, q2_norm, q3_norm]
    q_raw_true = data[:, 8:11]  # [q1_raw, q2_raw, q3_raw]
    return X_test, xyz_true, q_norm_true, q_raw_true


def wrapped_angle_diff(theta_true, theta_pred):
    """
    Coursework Task 4.A Revolute Joint Error Formula:
        delta_theta = atan2(sin(theta_true - theta_pred), cos(theta_true - theta_pred))
    """
    diff = theta_true - theta_pred
    return np.arctan2(np.sin(diff), np.cos(diff))


def evaluate_model_on_arm(model, arm_name, X_test, xyz_true, q_raw_true):
    """
    Executes steps 1-5 of Task 4.A for a single (model, arm) pair.
    """
    limits = get_norm_limits(arm_name)
    fk_fn = ned_forward_kinematics if arm_name == "ned" else scara_forward_kinematics

    # 1-2. Predict normalised joints from [is_scara, is_ned, x, y, z]_true and denormalise
    q_norm_pred = model.predict(X_test)
    q_raw_pred = denormalise_joints(q_norm_pred, limits)

    # 3-4. Apply Forward Kinematics to predicted joint values to get (x, y, z)_pred
    xyz_pred = np.array([fk_fn(q) for q in q_raw_pred], dtype=float)

    # 5A. Position Error: Euclidean distance between (x,y,z)_pred and (x,y,z)_true
    pos_errors_m = np.linalg.norm(xyz_true - xyz_pred, axis=1)

    # 5B. Joint Errors per active joint
    joint_stats = {}
    if arm_name == "ned":
        # All 3 active joints on Ned are revolute: [joint_1, joint_2, joint_3]
        joint_names = ["joint_1 (rad)", "joint_2 (rad)", "joint_3 (rad)"]
        joint_types = ["revolute", "revolute", "revolute"]
    else:
        # SCARA: q1 and q2 are revolute, q4 (index 2) is prismatic
        joint_names = ["base_arm_motor (rad)", "arm_motor (rad)", "shaft_linear_motor (m^2)"]
        joint_types = ["revolute", "revolute", "prismatic"]

    for j_idx, (j_name, j_type) in enumerate(zip(joint_names, joint_types)):
        true_j = q_raw_true[:, j_idx]
        pred_j = q_raw_pred[:, j_idx]

        if j_type == "revolute":
            # Exact Task 4.A wrapped angular difference
            err_signed = wrapped_angle_diff(true_j, pred_j)
            err_abs = np.abs(err_signed)
            joint_stats[f"q{j_idx+1}"] = {
                "name": j_name,
                "type": "revolute",
                "mean": float(np.mean(err_signed)),
                "std": float(np.std(err_signed)),
                "abs_mean": float(np.mean(err_abs)),
                "abs_std": float(np.std(err_abs)),
            }
        else:
            # Exact Task 4.A prismatic error: e_d = (d_true - d_pred)^2
            err_sq = (true_j - pred_j) ** 2
            err_abs = np.abs(true_j - pred_j)
            joint_stats[f"q{j_idx+1}"] = {
                "name": j_name,
                "type": "prismatic",
                "mean": float(np.mean(err_sq)),
                "std": float(np.std(err_sq)),
                "abs_mean_m": float(np.mean(err_abs)),
                "abs_std_m": float(np.std(err_abs)),
            }

    return {
        "pos_mean_m": float(np.mean(pos_errors_m)),
        "pos_std_m": float(np.std(pos_errors_m)),
        "pos_mean_mm": float(np.mean(pos_errors_m) * 1000.0),
        "pos_std_mm": float(np.std(pos_errors_m) * 1000.0),
        "pos_max_mm": float(np.max(pos_errors_m) * 1000.0),
        "joints": joint_stats,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_cache = {arm: load_test_set(arm) for arm in ARMS}

    csv_rows = []
    full_json_results = {}

    print("=" * 115)
    print("TASK 4.A: Analytical Forward Kinematics Evaluation on Independent Test Sets (N=500 per arm)")
    print("=" * 115)

    for approach in APPROACHES:
        # Check if models for this approach exist
        first_model_path = os.path.join(MODEL_DIR, f"model_{approach}_small.pkl")
        if not os.path.exists(first_model_path):
            # Fallback check for standard naming 'model_small.pkl'
            if approach == "single_fixed" and os.path.exists(os.path.join(MODEL_DIR, "model_small.pkl")):
                model_prefix = "model"
            else:
                continue
        else:
            model_prefix = f"model_{approach}"

        full_json_results[approach] = {}
        print(f"\n>>> APPROACH: {approach.upper()} <<<")
        print(
            f"{'Size':<8} | {'Arm':<6} | {'Pos Error Mean ± Std (mm)':<28} | "
            f"{'Joint 1 Error (Mean ± Std)':<28} | {'Joint 2 Error (Mean ± Std)':<28} | {'Joint 3/4 Error (Mean ± Std)'}"
        )
        print("-" * 135)

        for size_name in SIZES:
            model_path = os.path.join(MODEL_DIR, f"{model_prefix}_{size_name}.pkl")
            if not os.path.exists(model_path):
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            full_json_results[approach][size_name] = {}

            for arm in ARMS:
                X_test, xyz_true, _, q_raw_true = test_cache[arm]
                res = evaluate_model_on_arm(model, arm, X_test, xyz_true, q_raw_true)
                full_json_results[approach][size_name][arm] = res

                j1 = res["joints"]["q1"]
                j2 = res["joints"]["q2"]
                j3 = res["joints"]["q3"]

                pos_str = f"{res['pos_mean_mm']:7.3f} ± {res['pos_std_mm']:7.3f} mm"
                j1_str = f"{j1['mean']:+7.4f} ± {j1['std']:6.4f} (|e|={j1['abs_mean']:.4f})"
                j2_str = f"{j2['mean']:+7.4f} ± {j2['std']:6.4f} (|e|={j2['abs_mean']:.4f})"
                if arm == "ned":
                    j3_str = f"{j3['mean']:+7.4f} ± {j3['std']:6.4f} (|e|={j3['abs_mean']:.4f})"
                else:
                    j3_str = f"{j3['mean']:.2e} ± {j3['std']:.2e} m^2"

                print(f"{size_name:<8} | {arm.upper():<6} | {pos_str:<28} | {j1_str:<28} | {j2_str:<28} | {j3_str}")

                csv_rows.append({
                    "approach": approach,
                    "model_size": size_name,
                    "arm": arm,
                    "pos_mean_m": res["pos_mean_m"],
                    "pos_std_m": res["pos_std_m"],
                    "pos_mean_mm": res["pos_mean_mm"],
                    "pos_std_mm": res["pos_std_mm"],
                    "pos_max_mm": res["pos_max_mm"],
                    "q1_err_mean": j1["mean"],
                    "q1_err_std": j1["std"],
                    "q1_abs_mean": j1["abs_mean"],
                    "q2_err_mean": j2["mean"],
                    "q2_err_std": j2["std"],
                    "q2_abs_mean": j2["abs_mean"],
                    "q3_err_mean": j3["mean"],
                    "q3_err_std": j3["std"],
                    "q3_abs_mean": j3.get("abs_mean", j3.get("abs_mean_m")),
                })

    csv_path = os.path.join(RESULTS_DIR, "task_4a_summary.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    json_path = os.path.join(RESULTS_DIR, "task_4a_summary.json")
    with open(json_path, "w") as f:
        json.dump(full_json_results, f, indent=2)

    print("\n" + "=" * 115)
    print(f"Saved Task 4.A CSV summary  -> {csv_path}")
    print(f"Saved Task 4.A JSON details -> {json_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()