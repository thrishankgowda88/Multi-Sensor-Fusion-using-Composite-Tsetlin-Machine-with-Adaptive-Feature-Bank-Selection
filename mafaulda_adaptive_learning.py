import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
import random
from sklearn.utils import shuffle
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
import numpy as np
from statistics import mean, stdev
from scipy.stats import linregress
import os
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer



parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=int, required=True, help="channel")
parser.add_argument("--clause", type=int,default=30, help="number of clauses")
parser.add_argument("--T", type=int, default=22, help="value of threshold")
parser.add_argument("--s", type=float, default=3, help="value of specificity")
parser.add_argument("--bits", type=int, default=3, help="max_bits_per_feature")
parser.add_argument("--epochs", type=int, default=140, help="number of epochs in training phase")
args = parser.parse_args()


method = 'avg_acc'
S = args.s
number_of_clauses = args.clause
print(f'number_of_clause - {number_of_clauses}')
T = int(number_of_clauses * 0.75)
max_bits = args.bits

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
        feature_negation=feature_negation
    )


def process_csv_to_dict(csv_file, test_size=0.2, random_state=42):

    # Load the CSV file
    df = pd.read_csv(csv_file)
    columns_to_drop = ['filename','channel','segment','label_str','label']
    # Separate features and target variable
    cols_to_drop = [col for col in df.columns if col.endswith(('min', 'max', 'median'))]
    #df_cleaned = df.drop(columns=cols_to_drop)
    columns = cols_to_drop + columns_to_drop
    X = df.drop(columns=columns)  # Features
    Y = df['label']  # Labels
    

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print( f'shape {X_train.shape} , {X_test.shape} ')

    # binarize with current max_bits
    binarizer = StandardBinarizer(max_bits_per_feature=max_bits)
    binarizer.fit(X_train.to_numpy())
    data_dict = {
        "x_train": binarizer.transform(X_train.to_numpy()).astype(np.uint32),
        "x_test": binarizer.transform(X_test.to_numpy()).astype(np.uint32),
        "y_train": y_train.to_numpy().astype(np.uint32),
        "y_test": y_test.to_numpy().astype(np.uint32),
    }
    
    return data_dict


mfcc_dict = process_csv_to_dict(f"time_to_frequency/mfcc/channel_{args.channel}.csv")
spectral_dict = process_csv_to_dict(f"time_to_frequency/sd/channel_{args.channel}.csv")
stft_dict = process_csv_to_dict(f"time_to_frequency/stft/channel_{args.channel}.csv")


all_equal_train = (
    np.array_equal(mfcc_dict["y_train"], spectral_dict["y_train"]) and   
    np.array_equal(spectral_dict["y_train"], stft_dict["y_train"])
)

if not all_equal_train:
    raise ValueError("Mismatch in y_train labels across datasets!")
else: 
    print("train value true")


all_equal_test =( np.array_equal(mfcc_dict["y_test"], spectral_dict["y_test"]) and   
                  np.array_equal(spectral_dict["y_test"], stft_dict["y_test"]) )

if not all_equal_test:
    raise ValueError("Mismatch in y_test labels across datasets!")
else: 
    print("test value true")


all_equal = ( np.array_equal(mfcc_dict["x_train"], spectral_dict["x_train"]) and   
              np.array_equal(spectral_dict["x_train"], stft_dict["x_train"]) )

if all_equal:
    raise ValueError("Mismatch in x_train labels across datasets!")


import json
# This will store all results per epoch
training_log = {}
epoch = 120
batch_size = 200
# --- Batch generator ---
def batch_data(X, Y, batch_size):
    X, Y = shuffle(X, Y, random_state=42)
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], Y[i:i + batch_size]



adaptive_learning_log = {}

data_dicts = {
                'stft':stft_dict,
                'spectral' : spectral_dict,
                 'mfcc' : mfcc_dict



}


dict_names = list(data_dicts.keys())
random.seed(42)
random.shuffle(dict_names)
adaptive_learning_log['names'] = dict_names


dict_max_acc = {}
dict_last_tm_state = {}
dict_batch_accs = {name: [] for name in dict_names}

batch_num=0

from statistics import mean

# Collect per-dict accuracy list
dict_avg_acc = {}
dict_acc_list = []

#change this value in case you want more training window for each feature bank in the featurebank selection phase
B = 1

tm = tm_classifier(number_of_clauses=number_of_clauses,T=T,s=S)
# First pass: Train on all feature sets
for name in dict_names:
    for i in range(B):
        data = data_dicts[name]
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
        print(f"\nTraining on {name.upper()} in batches:")
        max_acc = 0
        acc_overbatch =[]   
        for i, (xb, yb) in enumerate(batch_data(x_train, y_train, batch_size)):
            batch_data_log = {}
    
            tm.fit(xb, yb)
            tm_test_pred , tm_test_scores = tm.predict(x_test , return_class_sums=True)
            acc = 100 * (tm_test_pred == y_test).mean()
    
            acc_overbatch.append(acc)
            batch_num=batch_num+1
    
            dict_avg_acc[name] = mean(acc_overbatch)
            dict_batch_accs[name].append(acc)
    
            
            batch_data_log = {
                "dict_name" : f'{name}',
                "batch_num" : batch_num,
                "y_pred": tm_test_pred.tolist(),
                "scores": tm_test_scores.tolist(),
                "acc": acc,
                }
            # max_acc = max(max_acc, acc)
            adaptive_learning_log[f'batch_{batch_num}'] = batch_data_log
        print(f"{name.upper()} Batch {i+1}: Test Accuracy = {acc:.2f}%")



# === Extended Metrics Computation ===
dict_metrics = {}
for name, acc_list in dict_batch_accs.items():
    acc_array = np.array(acc_list)

    #AvgAcc values
    avg = np.mean(acc_array)
    
    # Save Metrics
    dict_metrics[name] = {
        'avg_acc': avg
    }

best_dict_name = max(dict_metrics, key=lambda k: dict_metrics[k]['avg_acc'])

adaptive_learning_log['avg_acc'] = dict_metrics

print(f"\n Best feature set: {best_dict_name.upper()} with AUAC value { dict_metrics[best_dict_name]['avg_acc']:.2f}")






adaptive_learning_log['best_dict'] = best_dict_name

# Continue training for 7 more batches using the best feature set
print(f"\n Continuing training on {best_dict_name.upper()}:")

data = data_dicts[best_dict_name]
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']

for i in range(args.epochs):
        batch_data_log = {}
        tm.fit(x_train, y_train)
        tm_test_pred , tm_test_scores = tm.predict(x_test , return_class_sums=True)
        acc = 100 * (tm_test_pred == y_test).mean()
        batch_num=batch_num+1
        batch_data_log = {
            "dict_name" : f'{best_dict_name}',
            "batch_num" : batch_num,
            "y_pred": tm_test_pred.tolist(),
            "scores": tm_test_scores.tolist(),
            "acc": acc,
            }
        adaptive_learning_log[f'batch_{batch_num}'] = batch_data_log
        adaptive_learning_log['y_test_learning'] =   y_test.tolist()
        print(f"{best_dict_name.upper()} Continue Batch {batch_num}: Test Accuracy = {acc:.2f}%")


import os
# Define the directory
dir_name = "avg_acc"

# Create the directory if it doesn't exist
os.makedirs(dir_name, exist_ok=True)

with open(f"{dir_name}/channel_{args.channel}_{method}_cl_{number_of_clauses}_bits_{max_bits}_B_{B}.json", "w") as f:
    json.dump(adaptive_learning_log, f, indent=2)
