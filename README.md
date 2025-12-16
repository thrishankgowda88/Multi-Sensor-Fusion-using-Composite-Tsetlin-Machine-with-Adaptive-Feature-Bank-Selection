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

This script operates in **two sequential phases**:

### 1. Feature-Bank Selection Phase
- Performs adaptive selection of feature banks.
- The duration of this phase is controlled by the parameter **`B`**.
- **`B` is currently hard-coded inside the script**.
- To increase the feature-bank selection duration, manually update the value of `B` in the code.
- *(Future update)*: `B` will be exposed as a command-line argument.

### 2. Training Phase
- Trains a Tsetlin Machine using the selected feature bank(s).
- Supports configurable clauses, threshold, specificity, bit-resolution, and epochs.

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
