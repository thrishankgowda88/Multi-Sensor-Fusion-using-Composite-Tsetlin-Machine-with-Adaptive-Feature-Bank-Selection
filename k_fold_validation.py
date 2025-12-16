import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse
from datetime import datetime
import random
from sklearn.utils import shuffle
from statistics import mean, stdev
from scipy.stats import linregress
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
import json
import os
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=int, required=True, help="channel")
parser.add_argument("--clause", type=int, default=30, help="number of clauses")
parser.add_argument("--T", type=int, default=22, help="value of threshold")
parser.add_argument("--s", type=float, default=3, help="value of specificity")
parser.add_argument("--bits", type=int, default=3, help="")
parser.add_argument("--kfolds", type=int, default=5, help="number of folds for cross-validation")
args = parser.parse_args()

S = args.s
number_of_clauses = args.clause
print(f'number_of_clauses - {number_of_clauses}')
T = int(number_of_clauses * 0.75)  # you can still override with args.T if you want
max_bits = args.bits
k_folds = args.kfolds

def tm_classifier(
    number_of_clauses=30,
    T=22,
    s=3.0,
    platform="CPU",
    weighted_clauses=True,
    feature_negation=True
):
    return TMClassifier(
        number_of_clauses=number_of_clauses,
        T=T,
        s=s,
        platform=platform,
        weighted_clauses=weighted_clauses,
        feature_negation=feature_negation,
        seed=42
    )

def load_csv_features(csv_file):
    """
    Load full CSV (no train/test split here).
    We will let K-fold decide train/test per fold.
    """
    df = pd.read_csv(csv_file)

    columns_to_drop = ['filename', 'channel', 'segment', 'label_str', 'label']
    cols_to_drop = [col for col in df.columns if col.endswith(('min', 'max', 'median'))]
    columns = cols_to_drop + columns_to_drop

    X = df.drop(columns=columns).to_numpy()
    y = df['label'].to_numpy().astype(np.uint32)

    print(f"{csv_file} -> X shape {X.shape}, y shape {y.shape}")
    return X, y

# --- Load raw (continuous) features for all feature banks ---
mfcc_X, y_mfcc      = load_csv_features(f"time_to_frequency/mfcc/channel_{args.channel}.csv")
spectral_X, y_spec  = load_csv_features(f"time_to_frequency/sd/channel_{args.channel}.csv")
stft_X, y_stft      = load_csv_features(f"time_to_frequency/stft/channel_{args.channel}.csv")

# Sanity check: labels must match across feature representations
if not (np.array_equal(y_mfcc, y_spec) and np.array_equal(y_spec, y_stft)):
    raise ValueError("Mismatch in labels across feature sets!")
else:
    print("Label vectors match across all feature sets.")

y_all = y_mfcc  # common label vector

# Store raw feature matrices in a dict
data_X_dicts = {
    'mfcc': mfcc_X,
    'spectral': spectral_X,
    'stft': stft_X
}

batch_size = 200

# --- Batch generator ---
def batch_data(X, Y, batch_size):
    X, Y = shuffle(X, Y, random_state=42)
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], Y[i:i + batch_size]

# ========= K-FOLD CROSS-VALIDATION =========
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

all_folds_log = {}
fold_accuracies = []

print(f"\nStarting {k_folds}-fold cross-validation...\n")

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(mfcc_X, y_all), start=1):
    print(f"\n===== Fold {fold_idx}/{k_folds} =====")
    adaptive_learning_log = {}
    adaptive_learning_log['fold'] = fold_idx

    # ---- Create binarized data dicts for this fold ----
    data_dicts = {}
    for name, X_full in data_X_dicts.items():
        binarizer = StandardBinarizer(max_bits_per_feature=max_bits)
        binarizer.fit(X_full[train_idx])

        x_train_bin = binarizer.transform(X_full[train_idx]).astype(np.uint32)
        x_test_bin  = binarizer.transform(X_full[test_idx]).astype(np.uint32)

        data_dicts[name] = {
            "x_train": x_train_bin,
            "x_test": x_test_bin,
            "y_train": y_all[train_idx].astype(np.uint32),
            "y_test": y_all[test_idx].astype(np.uint32),
        }

    # Random order of feature banks
    dict_names = list(data_dicts.keys())
    random.shuffle(dict_names)
    adaptive_learning_log['names'] = dict_names

    # For feature bank selection phase
    dict_batch_accs = {name: [] for name in dict_names}
    dict_avg_acc = {}

    # One TM for this fold (keeps training across feature banks, like your original code)
    tm = tm_classifier(number_of_clauses=number_of_clauses, T=T, s=S)

    batch_num = 0

    # ========= PHASE 1: FEATURE BANK SELECTION =========
    for name in dict_names:
        data = data_dicts[name]
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']

        print(f"\n[Fold {fold_idx}] Training on {name.upper()} in batches (Feature Bank Selection):")
        acc_overbatch = []

        for i, (xb, yb) in enumerate(batch_data(x_train, y_train, batch_size)):
            batch_data_log = {}

            tm.fit(xb, yb)
            tm_test_pred, tm_test_scores = tm.predict(x_test, return_class_sums=True)
            acc = 100 * (tm_test_pred == y_test).mean()

            acc_overbatch.append(acc)
            batch_num += 1

            dict_avg_acc[name] = mean(acc_overbatch)
            dict_batch_accs[name].append(acc)

            batch_data_log = {
                "fold": fold_idx,
                "dict_name": f'{name}',
                "batch_num": batch_num,
                "y_pred": tm_test_pred.tolist(),
                "scores": tm_test_scores.tolist(),
                "acc": acc,
            }
            adaptive_learning_log[f'batch_{batch_num}'] = batch_data_log

        print(f"[Fold {fold_idx}] {name.upper()} - Last Batch {i+1}: Test Accuracy = {acc:.2f}%")

    # === Compute AUAC and select best feature bank for this fold ===
    dict_metrics = {}
    for name, acc_list in dict_batch_accs.items():
        acc_array = np.array(acc_list)
        auac = np.trapz(acc_array)  # Area under accuracy curve
        dict_metrics[name] = {
            'AUAC': auac
        }

    best_dict_name = max(dict_metrics, key=lambda k: dict_metrics[k]['AUAC'])
    adaptive_learning_log['AUAC'] = dict_metrics
    adaptive_learning_log['best_dict'] = best_dict_name

    print(f"\n[Fold {fold_idx}] Best feature set: {best_dict_name.upper()} "
          f"with AUAC = {dict_metrics[best_dict_name]['AUAC']:.2f}")

    # ========= PHASE 2: TRAINING PHASE ON BEST FEATURE BANK =========
    print(f"\n[Fold {fold_idx}] Continuing training on {best_dict_name.upper()} (Training Phase):")

    data = data_dicts[best_dict_name]
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    final_acc = 0.0

    # Same as your "for i in range(140)" loop
    for i in range(140):
        batch_data_log = {}

        tm.fit(x_train, y_train)
        tm_test_pred, tm_test_scores = tm.predict(x_test, return_class_sums=True)
        acc = 100 * (tm_test_pred == y_test).mean()
        final_acc = acc

        batch_num += 1
        batch_data_log = {
            "fold": fold_idx,
            "dict_name": f'{best_dict_name}',
            "batch_num": batch_num,
            "y_pred": tm_test_pred.tolist(),
            "scores": tm_test_scores.tolist(),
            "acc": acc,
        }
        adaptive_learning_log[f'batch_{batch_num}'] = batch_data_log
        adaptive_learning_log['y_test_learning'] = y_test.tolist()

        print(f"[Fold {fold_idx}] {best_dict_name.upper()} - Continue Batch {batch_num}: "
              f"Test Accuracy = {acc:.2f}%")

    print(f"\n[Fold {fold_idx}] FINAL Test Accuracy ({best_dict_name.upper()}): {final_acc:.2f}%")
    fold_accuracies.append(final_acc)
    all_folds_log[f'fold_{fold_idx}'] = adaptive_learning_log

# ========= SUMMARY ACROSS FOLDS =========
mean_acc = float(np.mean(fold_accuracies))
std_acc = float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0

print("\n===== K-FOLD CROSS-VALIDATION SUMMARY =====")
for i, acc in enumerate(fold_accuracies, start=1):
    print(f"Fold {i}: {acc:.2f}%")
print(f"\nMean Accuracy over {k_folds} folds: {mean_acc:.2f}%")
print(f"Std Dev of Accuracy: {std_acc:.2f}%")

all_folds_log['fold_accuracies'] = fold_accuracies
all_folds_log['mean_accuracy'] = mean_acc
all_folds_log['std_accuracy'] = std_acc

# ========= SAVE K-FOLD LOGS IN A SEPARATE FOLDER =========


# Root folder for all k-fold logs
log_root = "kfold_logs"

# Subfolder per channel and per k (e.g., kfold_logs/channel_1/)
log_dir = os.path.join(log_root, f"channel_{args.channel}")

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Store meta-info inside the log as well
all_folds_log['k_folds'] = k_folds
all_folds_log['channel'] = args.channel

# File name that encodes batch size, clauses, bits, and k
log_filename = (
    f"bs_{batch_size}_clauses_{number_of_clauses}_"
    f"bits_{max_bits}_k{k_folds}.json"
)

output_path = os.path.join(log_dir, log_filename)

with open(output_path, "w") as f:
    json.dump(all_folds_log, f, indent=2)

print(f"\nSaved detailed K-fold log to: {output_path}")

