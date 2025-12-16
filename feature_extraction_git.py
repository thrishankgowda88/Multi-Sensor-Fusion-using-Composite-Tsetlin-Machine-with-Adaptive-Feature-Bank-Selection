import os
import argparse
import warnings
import librosa
import numpy as np
import pandas as pd

# -----------------------------
# Defaults (same as your code)
# -----------------------------
SR_DEFAULT = 50000
SEGMENTS_DEFAULT = 10
FEATURE_STATS = ["mean", "median", "min", "max"]
CHANNELS_TO_KEEP_DEFAULT = [0, 1, 4, 7]
OUTPUT_ROOT_DEFAULT = "time_to_frequency"


# Helpers
def compute_statistics(vec):
    return [np.mean(vec), np.median(vec), np.min(vec), np.max(vec)]

def generate_feature_names(base, dim):
    return [f"{base}_{i}_{stat}" for i in range(dim) for stat in FEATURE_STATS]

def stats_block(feature, label, dim):
    # feature: (n_features, n_frames)
    # returns dict: {label_i_stat: value}
    return dict(
        zip(
            generate_feature_names(label, dim),
            [stat for i in range(feature.shape[0]) for stat in compute_statistics(feature[i])],
        )
    )

def process_segment(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    return {
        "mfcc": stats_block(mfcc, "mfcc", 12),
        "stft": stats_block(chroma_stft, "stft", 12),
        "sd": {
            **stats_block(rmse, "rmse", 1),
            **stats_block(zcr, "zcr", 1),
            **stats_block(centroid, "centroid", 1),
            **stats_block(bandwidth, "bandwidth", 1),
            **stats_block(rolloff, "rolloff", 1),
            **stats_block(contrast, "contrast", 7),
        },
    }

def iter_class_dirs(root_dir):
    class_labels = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )
    return class_labels

def process_csv_file(csv_path, label_str, label_int, sr, segments, channels_to_keep):
    df = pd.read_csv(csv_path, header=None)
    n_samples = df.shape[0]
    if n_samples < segments:
        raise ValueError(f"Not enough samples ({n_samples}) for segments={segments}")

    samples_per_segment = n_samples // segments
    basename = os.path.basename(csv_path)

    feature_blocks = {"mfcc": [], "stft": [], "sd": []}

    n_channels = df.shape[1]
    for ch in range(n_channels):
        if ch not in channels_to_keep:
            continue

        signal = df[ch].values.astype(float)

        for seg_idx in range(segments):
            start = seg_idx * samples_per_segment
            end = (seg_idx + 1) * samples_per_segment
            segment = signal[start:end]

            feats = process_segment(segment, sr)

            for key in feature_blocks:
                row = {
                    "filename": basename,
                    "channel": ch,
                    "segment": seg_idx,
                    "label_str": label_str,
                    "label": label_int,
                }
                row.update(feats[key])
                feature_blocks[key].append(row)

    return feature_blocks

def append_blocks(dst, src):
    for k in dst:
        dst[k].extend(src[k])

def save_channel_csvs(all_features, output_root, channels_to_keep):
    # Creates:
    # output_root/mfcc/channel_0.csv, ...
    for bank, rows in all_features.items():
        bank_dir = os.path.join(output_root, bank)
        os.makedirs(bank_dir, exist_ok=True)

        bank_df = pd.DataFrame(rows)
        if bank_df.empty:
            print(f"[WARN] No rows collected for feature bank '{bank}'. Nothing to save.")
            continue

        if "channel" not in bank_df.columns:
            raise ValueError(f"[BUG] Missing 'channel' column in collected rows for '{bank}'.")

        for ch in channels_to_keep:
            ch_df = bank_df[bank_df["channel"] == ch]
            out_path = os.path.join(bank_dir, f"channel_{ch}.csv")
            ch_df.to_csv(out_path, index=False)
            print(f"[OK] Saved: {out_path} (rows={len(ch_df)})")

def process_all_csv(root_dir, output_root, sr, segments, channels_to_keep):
    all_features = {"mfcc": [], "stft": [], "sd": []}

    class_labels = iter_class_dirs(root_dir)
    print("Class labels:", class_labels)

    for idx, label in enumerate(class_labels):
        print(f"\nProcessing -> Label: {label}, Numeric Label: {idx}")
        class_path = os.path.join(root_dir, label)

        for file in os.listdir(class_path):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(class_path, file)

            try:
                feats = process_csv_file(
                    file_path,
                    label_str=label,
                    label_int=idx,
                    sr=sr,
                    segments=segments,
                    channels_to_keep=channels_to_keep,
                )
                append_blocks(all_features, feats)
                print(f"  [OK] Processed: {file_path}")
            except Exception as e:
                print(f"  [ERR] {file_path} -> {e}")

    # Only final required outputs
    save_channel_csvs(all_features, output_root, channels_to_keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_dir",
        required=True,
        help='Root directory like "fault_database_10_class" containing class subfolders with CSV files.',
    )
    parser.add_argument("--sr", type=int, default=SR_DEFAULT)
    parser.add_argument("--segments", type=int, default=SEGMENTS_DEFAULT)
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=CHANNELS_TO_KEEP_DEFAULT,
        help="Channels to keep, e.g. --channels 0 1 4 7",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_ROOT_DEFAULT,
        help='Output root directory, default "time_to_ferquency"',
    )

    args = parser.parse_args()

    # Warnings (same as your notebook intent)
    warnings.filterwarnings("ignore", message="n_fft=1024 is too large")
    warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")
    warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set.")

    if not os.path.isdir(args.input_data_dir):
        raise ValueError(f"--input_data_dir does not exist or is not a directory: {args.input_data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    process_all_csv(
        root_dir=args.input_data_dir,
        output_root=args.output_dir,
        sr=args.sr,
        segments=args.segments,
        channels_to_keep=args.channels,
    )


if __name__ == "__main__":
    main()
