import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Times New Roman"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Wide CSV from run_composite_4ch.py")
    parser.add_argument("--B", type=int, required=True, help="B value (used only for phase-1 tick labeling)")
    parser.add_argument("--compress_factor", type=float, default=0.35, help="Shrink phase-1 on x-axis")
    parser.add_argument("--markevery", type=int, default=10, help="Marker spacing")
    parser.add_argument("--out_dir", type=str, default="plots", help="Output directory")
    parser.add_argument("--out_name", type=str, default=None, help="Output filename (pdf). If None, auto-name.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Required columns
    if "phase" not in df.columns or "step_num" not in df.columns:
        raise ValueError("CSV must contain columns: phase, step_num")

    # Identify accuracy columns (everything except phase, step_num)
    acc_cols = [c for c in df.columns if c not in ["phase", "step_num"]]
    if not acc_cols:
        raise ValueError("No accuracy columns found. Expected columns like s0,s1,s4,s7,s0+s1+s4+s7")

    # Ensure ordering: batch first then epoch
    phase_order = {"batch": 0, "epoch": 1}
    df["_phase_order"] = df["phase"].map(phase_order).fillna(99).astype(int)
    df = df.sort_values(by=["_phase_order", "step_num"]).drop(columns=["_phase_order"]).reset_index(drop=True)

    # Breakpoint: last batch step_num
    batch_df = df[df["phase"] == "batch"].copy()
    epoch_df = df[df["phase"] == "epoch"].copy()

    if batch_df.empty:
        raise ValueError("No 'batch' phase rows found in CSV.")
    break_point = int(batch_df["step_num"].max())  # number of batch steps

    # Build a continuous global x:
    # batch: 1..break_point
    # epoch: break_point + 1 .. break_point + epoch_steps
    x_global = []
    for _, r in df.iterrows():
        if r["phase"] == "batch":
            x_global.append(int(r["step_num"]))
        else:
            x_global.append(break_point + int(r["step_num"]))
    x_global = np.array(x_global, dtype=int)

    max_x = int(x_global.max())

    # Compression transform (same idea as your code)
    compress_factor = args.compress_factor

    def transform_x(x):
        if x <= break_point:
            return x * compress_factor
        else:
            return break_point * compress_factor + (x - break_point)

    tx = np.array([transform_x(x) for x in x_global])

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(4, 3))

    # Shaded Phase-1 (batch)
    plt.axvspan(
        transform_x(1),
        transform_x(break_point),
        facecolor="#e0f7fa",
        alpha=0.3
    )

    # Shaded Phase-2 (epoch)
    plt.axvspan(
        transform_x(break_point),
        transform_x(max_x),
        facecolor="#F4FACA",
        alpha=0.3
    )

    colors = ['r', 'g', '#905C7D', 'y', 'k']
    markers = ['o', 's', '^', 'd', '*']

    # Custom legend labels (same order as acc_cols in CSV)
    labels = ['Tachometer', 'Accelerometer-1', 'Accelerometer-2', 'Microphone', 'Composite-TM']

    # Safety: if CSV has unexpected number of accuracy columns, fall back to column names
    if len(acc_cols) != len(labels):
        print(f"[WARN] CSV has {len(acc_cols)} accuracy columns, but labels list has {len(labels)} items.")
        print("[WARN] Falling back to using CSV column names as labels.")
        labels = acc_cols

    for i, col in enumerate(acc_cols):
        plt.plot(
            tx,
            df[col].values.astype(float),
            label=labels[i],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markevery=args.markevery,
            markersize=5,
            linewidth=1
        )

    # Vertical separator at start of Phase-2 (compressed)
    plt.axvline(
        transform_x(break_point),
        color="gray",
        linestyle="--",
        linewidth=1
    )

    # -----------------------------
    # Custom ticks
    # -----------------------------
    phase1_ticks = list(range(0, break_point + 1, 20))
    if 0 not in phase1_ticks:
        phase1_ticks = [0] + phase1_ticks

    phase2_ticks = list(range(break_point + 20, max_x + 1, 20))

    xticks = phase1_ticks + phase2_ticks
    xticks_transformed = [transform_x(x) for x in xticks]

    phase1_end_label = (break_point // 20) * args.B

    xtick_labels = []
    for x in xticks:
        if x <= break_point:
            if x == 0:
                xtick_labels.append("0")
            else:
                xtick_labels.append(str((x // 20) * args.B))
        else:
            xtick_labels.append(str(phase1_end_label + (x - break_point)))

    plt.xticks(xticks_transformed, xtick_labels)

    # Axes
    plt.ylim(0, 100)
    plt.xlim(transform_x(0), transform_x(max_x))

    plt.xlabel("Epoch", fontsize=10, labelpad=0)
    plt.ylabel("Accuracy (%)", fontsize=10, labelpad=-3)

    plt.legend(fontsize=7, ncol=2, loc="lower right")

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = args.out_name
    if out_name is None:
        base = os.path.splitext(os.path.basename(args.csv_path))[0]
        out_name = f"{base}_B_{args.B}.pdf"
    out_path = os.path.join(args.out_dir, out_name)

    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.tight_layout()
    plt.show()

    print(f"[DONE] Plot saved -> {out_path}")
    print(f"[INFO] break_point(batch steps)={break_point}, max_x={max_x}, phase1_end_label={phase1_end_label}")


if __name__ == "__main__":
    main()
