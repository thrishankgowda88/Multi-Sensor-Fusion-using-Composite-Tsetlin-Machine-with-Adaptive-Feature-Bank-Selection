# Multi-Sensor-Fusion-using-Composite-Tsetlin-Machine-with-Adaptive-Feature-Bank-Selection

This repository presents a **Composite Tsetlin Machine (Composite-TM)** framework for **multi-sensor fusion** with an **Adaptive Feature Bank Selection** strategy, evaluated on the *MaFaulDa* dataset.  
The approach is designed to improve robustness and performance by dynamically selecting informative feature banks before performing sensor fusion.

---

## Key Contributions

- Adaptive feature-bank selection prior to model training
- Composite Tsetlin Machine for multi-sensor fusion
- Support for both **single-sensor** and **multi-sensor (composite)** experiments
- Reproducible experimental pipeline with plotting utilities

---

##Dataset used
The **MaFaulDa (Machinery Fault Database)** is a comprehensive dataset designed for the analysis and classification of machinery faults. It is widely used for developing and evaluating machine learning models for fault diagnosis in rotating machinery.

---

### Dataset Description

The MaFaulDa dataset consists of **1951 multivariate time-series signals** acquired from sensors mounted on **SpectraQuest’s Machinery Fault Simulator (MFS) Alignment–Balance–Vibration (ABVT)** platform.

The dataset includes **six different simulated operating conditions**:

- Normal operation  
- Imbalance fault  
- Horizontal misalignment fault  
- Vertical misalignment fault  
- Inner bearing fault  
- Outer bearing fault  

### Data Access and Usage

The MaFaulDa dataset can be downloaded from Kaggle:

- **Dataset link:** https://www.kaggle.com/datasets/vuxuancu/mafaulda-full

After downloading, extract the dataset and place the CSV files in the appropriate input directory as required by the Feature extraction pipline.

## Dataset Directory Structure and Class Definition

The MaFaulDa dataset is organized into a hierarchical directory structure, where fault types are grouped based on machine configuration and fault location. The actual directory structure is shown below:
```text
mafaulda/
├── horizontal-misalignment/
├── imbalance/
├── normal/
├── overhang/
│   ├── ball_fault/
│   ├── cage_fault/
│   └── outer_race/
├── underhang/
│   ├── ball_fault/
│   ├── cage_fault/
│   └── outer_race/
└── vertical-misalignment/

In this structure:
- `horizontal-misalignment`, `vertical-misalignment`, `imbalance`, and `normal` are top-level fault categories.
- `overhang` and `underhang` contain **sub-faults** related to bearing defects:
  - `ball_fault`
  - `cage_fault`
  - `outer_race`
```
---

## Class Reorganization (10-Class Setup)

In this work, **each sub-folder is treated as an independent class**, rather than grouping them under a parent category.  
This converts the original hierarchical structure into a **10-class classification problem**.
Store them in a dir to extract time to frequency features.

The resulting class definition is:

1. Normal  
2. Imbalance  
3. Horizontal misalignment  
4. Vertical misalignment  
5. Overhang – ball fault  
6. Overhang – cage fault  
7. Overhang – outer race fault  
8. Underhang – ball fault  
9. Underhang – cage fault  
10. Underhang – outer race fault 


## Feature Extraction

Time-to-frequency domain features are extracted using the `feature_extraction_git.py` script.

To extract the time-to-frequency features from the dataset, run:

```bash
python feature_extraction.py --input_data_dir <your_dir>
```


## Repository Contents

| File | Description |
|----|----|
| `mafaulda_adaptive_learning.py` | Adaptive feature-bank selection and single-sensor training |
| `composite_TM.py` | Composite Tsetlin Machine for multi-sensor fusion |
| `plot_composite_curve.py` | Utility for plotting single-sensor and composite accuracy curves |
| `avg_acc/` | Directory containing pre-processed input data and statistics |
| `single_and_all4.csv` | Example CSV file for plotting accuracy curves |

---

## `mafaulda_adaptive_learning.py`
Supports configurable clauses, threshold, specificity, bit-resolution, and
This script operates in **two sequential phases**:

### 1. Feature-Bank Selection Phase
- Performs adaptive selection of feature banks.
- The duration of this phase is controlled by the parameter **`B`**.
- **`B` is currently hard-coded inside the script**.
- To increase the feature-bank selection duration, manually update the value of `B` in the code.
- *(Future update)*: `B` will be exposed as a command-line argument.

### 2. Training Phase
- Trains a Tsetlin Machine using the selected feature bank(s).
- The number of epochs to be trained in the training phase is decided by **`epochs`**  .

---

## Sensor Channel Mapping

The following mapping is used throughout all experiments:

| Channel ID | Sensor |
|---------|--------|
| `0` | Tachometer |
| `1` | Accelerometer-1 |
| `4` | Accelerometer-2 |
| `7` | Microphone |

---

## Running Single-Sensor Experiments

Use the following command format:

```bash
python mafaulda_adaptive_learning.py \
    --channel 7 \
    --clause 30 \
    --T 22 \
    --s 3.0 \
    --bits 3 \
    --epochs 140 \
    --input_dir avg_acc
```

## Running Composite Multi-Sensor Fusion

After completing the single-sensor adaptive learning runs, composite multi-sensor fusion is performed using the Composite Tsetlin Machine. This step combines information from multiple sensors.

To run the composite fusion, execute:

```bash
python composite_TM.py
```

## Plotting Accuracy Curves

The accuracy curves for single-sensor models and the composite multi-sensor model are generated using the `plot_composite_curve.py` script. This script reads the stored accuracy values from a CSV file and visualizes the performance across training phases.

To generate the accuracy curves, run:

```bash
python plot_composite_curve.py --csv_path single_and_all4.csv --B 1
```


