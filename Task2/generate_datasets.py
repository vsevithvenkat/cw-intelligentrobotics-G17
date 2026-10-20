# generate_datasets.py
import os
import csv
import numpy as np

from ned_fk import (
    ned_forward_kinematics,
    sample_ned_collision_free,
)
from scara_fk import (
    scara_forward_kinematics,
    sample_scara_collision_free,
)


DATASET_SIZES = {
    "small":  1000,   # 800 train,   200 val  (~9.3 pts/axis  - sparse baseline)
    "medium": 5000,   # 4000 train,  1000 val (~15.9 pts/axis - intermediate)
    "large":  25000,  # 20000 train, 5000 val (~27.1 pts/axis - high-density asymptotic)
}

TRAIN_RATIO = 0.80
TEST_SET_SIZE = 1000
TASK_4B_TEST_SIZE = 100


USE_ELBOW_UP_BOUNDS_FOR_NORM = True

NOMINAL_NED_LIMITS = np.array([
    [-2.80, 2.80],  # q1: joint_1 (rad)
    [-0.80, 0.80],  # q2: joint_2 (rad)
    [ 0.00, 1.30],  # q3: joint_3 (rad) [Elbow-up >= 0.0]
], dtype=float)

NOMINAL_SCARA_LIMITS = np.array([
    [-0.73, 0.73],  # q1: base_arm_motor (rad)
    [ 0.00, 0.83],  # q2: arm_motor (rad) [Elbow-up >= 0.0]
    [-0.20, 0.00],  # q4: shaft_linear_motor (m) [Nominal datasheet limit]
], dtype=float)

# [DECISION 3] Fixed random seeds for 100% reproducibility across group & markers
TRAIN_VAL_SEED = 2026
TEST_SEED = 99999

OUTPUT_DIR = "datasets"


# ==============================================================================
# NORMALISATION & DENORMALISATION FUNCTIONS (TASK 2 & TASK 3 FORMULAS)
# ==============================================================================

def get_norm_limits(arm_name):
    """
    Returns the nominal [q_min, q_max] array used for [-1, 1] min-max normalisation.
    Strictly adheres to Section 4 of Assumption.md.
    """
    if arm_name == "ned":
        limits = NOMINAL_NED_LIMITS.copy()
        if not USE_ELBOW_UP_BOUNDS_FOR_NORM:
            limits[2, 0] = -1.30
        return limits
    elif arm_name == "scara":
        limits = NOMINAL_SCARA_LIMITS.copy()
        if not USE_ELBOW_UP_BOUNDS_FOR_NORM:
            limits[1, 0] = -0.83
        return limits
    else:
        raise ValueError(f"Unknown arm_name: {arm_name}")


def normalise_joints(joints, limits):
    """
    Normalises joint values to [-1.0, 1.0] using the coursework formula:
        q_norm = 2 * (q - q_min) / (q_max - q_min) - 1
    """
    q = np.asarray(joints, dtype=float)
    q_min = limits[:, 0]
    q_max = limits[:, 1]
    return 2.0 * (q - q_min) / (q_max - q_min) - 1.0


def denormalise_joints(q_norm, limits):
    """
    De-normalises joint values from [-1.0, 1.0] back to physical units:
        q = q_min + ((q_norm + 1) / 2) * (q_max - q_min)
    """
    qn = np.asarray(q_norm, dtype=float)
    q_min = limits[:, 0]
    q_max = limits[:, 1]
    return q_min + ((qn + 1.0) / 2.0) * (q_max - q_min)


# ==============================================================================
# DATASET GENERATION & NESTED SPLITTING
# ==============================================================================

def generate_samples_for_arm(arm_name, n_samples, rng):
    """
    Generates random collision-free elbow-up joint configurations, computes
    their FK TCP (x, y, z) coordinates, and normalises the joint values.
    """
    limits = get_norm_limits(arm_name)

    if arm_name == "ned":
        # Sample within collision-free elbow-up workspace (z_TCP >= 0.12m, q2+q3 <= 1.15)
        q_raw = sample_ned_collision_free(n_samples, rng)
        fk_fn = ned_forward_kinematics
        flags = np.tile([0.0, 1.0], (n_samples, 1))  # [is_scara=0, is_ned=1]
    elif arm_name == "scara":
        # Sample within collision-free elbow-up workspace (q4 in [-0.17, 0.00] m)
        q_raw = sample_scara_collision_free(n_samples, rng)
        fk_fn = scara_forward_kinematics
        flags = np.tile([1.0, 0.0], (n_samples, 1))  # [is_scara=1, is_ned=0]
    else:
        raise ValueError(f"Unknown arm_name: {arm_name}")

    xyz = np.array([fk_fn(q) for q in q_raw], dtype=float)
    q_norm = normalise_joints(q_raw, limits)

    # Verify exact mathematical invertibility of normalisation
    q_reconstructed = denormalise_joints(q_norm, limits)
    assert np.allclose(q_raw, q_reconstructed, atol=1e-12), "Normalisation round-trip error!"
    assert np.all((q_norm >= -1.0) & (q_norm <= 1.0)), "Normalised values out of [-1, 1]!"

    return {
        "flags": flags,    # shape (N, 2): [is_scara, is_ned]
        "xyz": xyz,        # shape (N, 3): [x, y, z] in metres
        "q_norm": q_norm,  # shape (N, 3): [q1_norm, q2_norm, q3_norm] in [-1, 1]
        "q_raw": q_raw,    # shape (N, 3): physical units (rad / m)
    }


def slice_dataset(data_dict, start_idx, end_idx):
    """Slices all arrays in a dataset dictionary from start_idx to end_idx."""
    return {k: v[start_idx:end_idx] for k, v in data_dict.items()}


def save_csv(filepath, data_dict):
    """
    Saves dataset to CSV.
    Columns: is_scara, is_ned, x, y, z, q1_norm, q2_norm, q3_norm, q1_raw, q2_raw, q3_raw
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = [
        "is_scara", "is_ned",
        "x", "y", "z",
        "q1_norm", "q2_norm", "q3_norm",
        "q1_raw", "q2_raw", "q3_raw"
    ]
    rows = np.hstack([
        data_dict["flags"],
        data_dict["xyz"],
        data_dict["q_norm"],
        data_dict["q_raw"]
    ])

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ==============================================================================
# DATASET ANALYSIS & VERIFICATION MODULE (FOR TASK 2 & TASK 5 POSTER/VIDEO)
# ==============================================================================

def analyze_and_verify_datasets(arm_datasets):
    """
    Performs comprehensive statistical and structural verification of all
    generated datasets (nesting, elbow-up constraint, normalisation bounds,
    data leakage check, and workspace coordinate distributions).
    """
    print("\n" + "=" * 88)
    print("DATASET ANALYSIS & INTEGRITY VERIFICATION REPORT")
    print("=" * 88)

    for arm in ["ned", "scara"]:
        print(f"\n>>> ANALYSIS FOR {arm.upper()} ARM <<<")
        splits = arm_datasets[arm]

        # 1. Verify Strict Nesting (Small C Medium C Large)
        small_tr_xyz = splits["small_train"]["xyz"]
        med_tr_xyz = splits["medium_train"]["xyz"]
        large_tr_xyz = splits["large_train"]["xyz"]

        small_val_xyz = splits["small_val"]["xyz"]
        med_val_xyz = splits["medium_val"]["xyz"]
        large_val_xyz = splits["large_val"]["xyz"]

        train_nested_ok = (
            np.array_equal(small_tr_xyz, med_tr_xyz[:len(small_tr_xyz)]) and
            np.array_equal(med_tr_xyz, large_tr_xyz[:len(med_tr_xyz)])
        )
        val_nested_ok = (
            np.array_equal(small_val_xyz, med_val_xyz[:len(small_val_xyz)]) and
            np.array_equal(med_val_xyz, large_val_xyz[:len(med_val_xyz)])
        )

        # 2. Verify Elbow-Up Constraint (elbow joint >= 0)
        elbow_idx = 2 if arm == "ned" else 1
        elbow_name = "joint_3 (q3)" if arm == "ned" else "arm_motor (q2)"
        min_elbow_train = np.min(splits["large_train"]["q_raw"][:, elbow_idx])
        min_elbow_test = np.min(splits["test"]["q_raw"][:, elbow_idx])
        elbow_up_ok = (min_elbow_train >= 0.0) and (min_elbow_test >= 0.0)

        # 3. Verify Zero Data Leakage Between Train, Val, and Test
        test_xyz = splits["test"]["xyz"]
        diff_tr = np.linalg.norm(large_tr_xyz[:, None, :] - test_xyz[None, :, :], axis=-1)
        diff_val = np.linalg.norm(large_val_xyz[:, None, :] - test_xyz[None, :, :], axis=-1)
        exact_overlap_count = int(np.sum(diff_tr == 0.0) + np.sum(diff_val == 0.0))

        print(f"  [Check 1] Strict Train Nesting (Small C Med C Large) : {'PASS' if train_nested_ok else 'FAIL'}")
        print(f"  [Check 2] Strict Val Nesting   (Small C Med C Large) : {'PASS' if val_nested_ok else 'FAIL'}")
        print(f"  [Check 3] Elbow-Up Only ({elbow_name} >= 0)       : {'PASS' if elbow_up_ok else 'FAIL'} (min={min_elbow_train:.4f} rad)")
        print(f"  [Check 4] Test Set Independence (0 overlaps)         : {'PASS' if exact_overlap_count == 0 else 'FAIL'} (overlaps={exact_overlap_count})")

        # 4. Statistical Summary Table for Large Train & Test Sets
        n_large_tr = len(large_tr_xyz)
        n_test = len(test_xyz)
        for label, key in [(f"LARGE TRAIN (N={n_large_tr})", "large_train"), (f"TEST SET (N={n_test})", "test")]:
            d = splits[key]
            xyz = d["xyz"]
            qn = d["q_norm"]
            qr = d["q_raw"]

            print(f"\n  --- {arm.upper()} {label} SUMMARY STATISTICS ---")
            print(f"  {'Variable':<14} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
            print("  " + "-" * 64)
            for i, axis in enumerate(["x (m)", "y (m)", "z (m)"]):
                print(f"  {axis:<14} | {np.min(xyz[:, i]):10.4f} | {np.max(xyz[:, i]):10.4f} | {np.mean(xyz[:, i]):10.4f} | {np.std(xyz[:, i]):10.4f}")
            for i, jname in enumerate(["q1_raw", "q2_raw", "q3_raw"]):
                print(f"  {jname:<14} | {np.min(qr[:, i]):10.4f} | {np.max(qr[:, i]):10.4f} | {np.mean(qr[:, i]):10.4f} | {np.std(qr[:, i]):10.4f}")
            for i, jnorm in enumerate(["q1_norm", "q2_norm", "q3_norm"]):
                print(f"  {jnorm:<14} | {np.min(qn[:, i]):10.4f} | {np.max(qn[:, i]):10.4f} | {np.mean(qn[:, i]):10.4f} | {np.std(qn[:, i]):10.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_val_rng = np.random.default_rng(TRAIN_VAL_SEED)
    test_rng = np.random.default_rng(TEST_SEED)

    max_size = DATASET_SIZES["large"]
    n_large_train = int(max_size * TRAIN_RATIO)

    print("=" * 88)
    print("TASK 2: Generating Optimal Nested Training/Validation Datasets & Independent Test Sets")
    print("=" * 88)

    arm_datasets = {"ned": {}, "scara": {}}

    for arm in ["ned", "scara"]:
        print(f"\n--- Generating datasets for {arm.upper()} ---")

        # 1. Generate the master pool for the LARGE dataset first
        large_pool = generate_samples_for_arm(arm, max_size, train_val_rng)

        # Split Large pool 80/20 into Large Train (20,000) and Large Val (5,000)
        large_train = slice_dataset(large_pool, 0, n_large_train)
        large_val = slice_dataset(large_pool, n_large_train, max_size)

        # 2. Slice strictly nested Small, Medium, and Large splits
        for size_name, total_n in DATASET_SIZES.items():
            n_train = int(total_n * TRAIN_RATIO)
            n_val = total_n - n_train
            grid_density = n_train ** (1.0 / 3.0)

            sub_train = slice_dataset(large_train, 0, n_train)
            sub_val = slice_dataset(large_val, 0, n_val)

            train_path = os.path.join(OUTPUT_DIR, f"{arm}_{size_name}_train.csv")
            val_path = os.path.join(OUTPUT_DIR, f"{arm}_{size_name}_val.csv")

            save_csv(train_path, sub_train)
            save_csv(val_path, sub_val)

            arm_datasets[arm][f"{size_name}_train"] = sub_train
            arm_datasets[arm][f"{size_name}_val"] = sub_val

            print(
                f"  [{size_name.upper():<6}] Total: {total_n:<5} | "
                f"Train: {n_train:<5} (~{grid_density:4.1f} pts/axis) | "
                f"Val: {n_val:<4}"
            )

        # 3. Generate independent Test Set (1000 samples for Task 4.A)
        test_data = generate_samples_for_arm(arm, TEST_SET_SIZE, test_rng)
        test_path = os.path.join(OUTPUT_DIR, f"{arm}_test.csv")
        save_csv(test_path, test_data)
        arm_datasets[arm]["test"] = test_data
        print(f"  [TEST4A] Total: {TEST_SET_SIZE:<5} -> Saved full test set ({test_path})")

        # 4. Also save the first 100 test samples explicitly for Task 4.B Webots evaluation
        test_4b_data = slice_dataset(test_data, 0, TASK_4B_TEST_SIZE)
        test_4b_path = os.path.join(OUTPUT_DIR, f"{arm}_test_100.csv")
        save_csv(test_4b_path, test_4b_data)
        print(f"  [TEST4B] Total: {TASK_4B_TEST_SIZE:<5} -> Saved first 100 test samples ({test_4b_path})")

    # Run automated dataset analysis and verification
    analyze_and_verify_datasets(arm_datasets)


if __name__ == "__main__":
    main()