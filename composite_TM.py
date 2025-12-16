# run_composite_4ch.py
#
# Runs:
#   python mafaulda_adaptive_learning.py --channel 0 ...
#   python mafaulda_adaptive_learning.py --channel 1 ...
#   python mafaulda_adaptive_learning.py --channel 4 ...
#   python mafaulda_adaptive_learning.py --channel 7 ...
#
# Then loads JSONs from avg_acc/ and saves ONE CSV with:
#   - single sensor performance: s0, s1, s4, s7
#   - composite (all 4): s0+s1+s4+s7
#
# Also prints progress + max-accuracy info to terminal (no extra summary rows in CSV).

import os
import re
import json
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def batch_sort_key(batch_key: str) -> int:
    return int(batch_key.split("_")[1])


def safe_num_feature_banks(avg_acc_obj) -> int:
    if avg_acc_obj is None:
        return 1
    if isinstance(avg_acc_obj, dict):
        return max(1, len(avg_acc_obj))
    if isinstance(avg_acc_obj, list):
        return max(1, len(avg_acc_obj))
    if isinstance(avg_acc_obj, (int, float)):
        return max(1, int(avg_acc_obj))
    return 1


def parse_B_from_filename(path: str) -> int:
    m = re.search(r"_B_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 1


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    smax = scores.max()
    smin = scores.min()
    return (scores - smin) / (smax - smin + 1e-8)


def accuracy_from_scores(scores: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(scores, axis=1)
    return 100.0 * float((preds == y_true).mean())


def find_channel_json(avg_acc_dir: str, ch: int, pattern: str = None) -> str:
    if pattern is None:
        patt = os.path.join(avg_acc_dir, f"channel_{ch}_avg_acc_*.json")
    else:
        patt = os.path.join(avg_acc_dir, pattern.format(ch=ch))

    matches = sorted(glob.glob(patt))
    if not matches:
        raise FileNotFoundError(f"No JSON found for channel {ch} using pattern: {patt}")
    return matches[-1]


def run_adaptive_script(
    python_exe: str,
    adaptive_script: str,
    ch: int,
    clause: int,
    T: int,
    s: float,
    bits: int,
    epochs: int,
    extra_args: list[str],
):
    cmd = [
        python_exe,
        adaptive_script,
        "--channel",
        str(ch),
        "--clause",
        str(clause),
        "--T",
        str(T),
        "--s",
        str(s),
        "--bits",
        str(bits),
        "--epochs",
        str(epochs),
    ] + extra_args

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] Finished adaptive learning for channel {ch}")


def split_phase_keys(ref_sdict: dict, total_batches: int):
    batch_keys = sorted(
        [k for k in ref_sdict.keys() if k.startswith("batch_")],
        key=batch_sort_key,
    )
    return batch_keys[:total_batches], batch_keys[total_batches:]


def common_keys(sensor_dicts: dict, keys: list[str]) -> list[str]:
    ok = []
    for k in keys:
        if all(k in sensor_dicts[sid] for sid in sensor_dicts):
            ok.append(k)
    return ok


def ensemble_scores_for_key(sensor_dicts: dict, combo_sids: list[str], key: str) -> np.ndarray:
    votes = None
    for sid in combo_sids:
        scores = np.array(sensor_dicts[sid][key]["scores"])
        ns = normalize_scores(scores)
        votes = ns if votes is None else (votes + ns)
    return votes


def print_max_info_wide(df_wide: pd.DataFrame, col: str, phase: str):
    sub = df_wide[df_wide["phase"] == phase]
    if sub.empty or col not in sub.columns:
        print(f"[WARN] No data for col={col} phase={phase}")
        return

    # ignore NaNs
    s = sub[col].dropna()
    if s.empty:
        print(f"[WARN] All NaN for col={col} phase={phase}")
        return

    idx = s.idxmax()
    best_row = df_wide.loc[idx]
    print(
        f"[MAX] {col:>14} | phase={phase:<5} | best_acc={best_row[col]:.2f}% | step_num={int(best_row['step_num'])}"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    # ---- runner inputs ----
    parser.add_argument("--avg_acc_dir", default="avg_acc")
    parser.add_argument("--adaptive_script", default="mafaulda_adaptive_learning.py")
    parser.add_argument("--python", default="python")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1, 4, 7])
    parser.add_argument("--output_csv", default="single_and_all4.csv")
    parser.add_argument(
        "--json_pattern",
        default=None,
        help='Optional filename pattern inside avg_acc_dir, must include "{ch}".',
    )
    parser.add_argument("--skip_run", action="store_true")

    # ---- forwarded args to adaptive learning ----
    parser.add_argument("--clause", type=int, default=30)
    parser.add_argument("--T", type=int, default=22)
    parser.add_argument("--s", type=float, default=3.0)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=140)

    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to mafaulda_adaptive_learning.py",
    )

    args = parser.parse_args()
    os.makedirs(args.avg_acc_dir, exist_ok=True)

    # 1) Run adaptive learning
    if not args.skip_run:
        for ch in args.channels:
            run_adaptive_script(
                python_exe=args.python,
                adaptive_script=args.adaptive_script,
                ch=ch,
                clause=args.clause,
                T=args.T,
                s=args.s,
                bits=args.bits,
                epochs=args.epochs,
                extra_args=args.extra_args,
            )
    else:
        print("[SKIP] Not running adaptive learning. Reading existing JSONs only.")

    # 2) Load JSONs
    channel_to_path = {}
    channel_to_dict = {}
    for ch in args.channels:
        jpath = find_channel_json(args.avg_acc_dir, ch, args.json_pattern)
        channel_to_path[ch] = jpath
        channel_to_dict[ch] = load_json(jpath)
        print(f"[LOAD] ch={ch} -> {jpath}")

    # Map to s0,s1,s4,s7 keys
    sensor_dicts = {f"s{ch}": channel_to_dict[ch] for ch in args.channels}
    single_cols = [f"s{ch}" for ch in args.channels]
    combo_col = "+".join(single_cols)

    # Ground truth
    first_sid = sorted(sensor_dicts.keys())[0]
    y_true = np.array(sensor_dicts[first_sid]["y_test_learning"])

    # 3) Infer phase split using your rule
    ref_ch = int(first_sid[1:])
    ref_path = channel_to_path[ref_ch]
    ref_dict = sensor_dicts[first_sid]

    num_feature_banks = safe_num_feature_banks(ref_dict.get("avg_acc", None))
    B_val = parse_B_from_filename(ref_path)

    BATCHES_PER_FEATUREBANK = 20
    total_batches = BATCHES_PER_FEATUREBANK * num_feature_banks * B_val

    print(f"[INFO] num_feature_banks={num_feature_banks}, B={B_val}, total_batches={total_batches}")

    batch_phase_keys, epoch_phase_keys = split_phase_keys(ref_dict, total_batches)
    batch_phase_keys = common_keys(sensor_dicts, batch_phase_keys)
    epoch_phase_keys = common_keys(sensor_dicts, epoch_phase_keys)

    print(f"[INFO] batch_phase_keys={len(batch_phase_keys)}, epoch_phase_keys={len(epoch_phase_keys)}")

    # 4) Build WIDE rows: one row per step_num with columns for each sensor and composite
    wide_rows = []

    def add_wide_rows(phase_name: str, keys: list[str]):
        for step_num, k in enumerate(keys, start=1):
            row = {
                "phase": phase_name,
                "step_num": step_num,
            }

            # single sensors
            for sid in single_cols:
                scores = np.array(sensor_dicts[sid][k]["scores"])
                row[sid] = accuracy_from_scores(scores, y_true)

            # composite all-4
            votes = ensemble_scores_for_key(sensor_dicts, single_cols, k)
            row[combo_col] = accuracy_from_scores(votes, y_true)

            wide_rows.append(row)

    add_wide_rows("batch", batch_phase_keys)
    add_wide_rows("epoch", epoch_phase_keys)

    df_wide = pd.DataFrame(wide_rows)

    # Ensure column order
    df_wide = df_wide[["phase", "step_num"] + single_cols + [combo_col]]

    # 5) Print max info to terminal
    print("\n========== MAX ACCURACY REPORT ==========")
    for c in single_cols + [combo_col]:
        print_max_info_wide(df_wide, c, "batch")
        print_max_info_wide(df_wide, c, "epoch")
    print("========================================\n")

    # 6) Save CSV
    df_wide.to_csv(args.output_csv, index=False)
    print(f"[DONE] Saved -> {args.output_csv} (rows={len(df_wide)})")


if __name__ == "__main__":
    main()
