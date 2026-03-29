# Linear Regression from Scratch

A NumPy-based implementation of multivariate linear regression trained via gradient descent — no scikit-learn model, just the math.

## Overview

This project builds a linear regression model from the ground up to predict **total hospital charges (`TOTCHG`)** for pediatric patients, using a real-world healthcare dataset. The goal is to understand the internals of gradient descent and model training without relying on high-level ML libraries.

## Dataset

**File:** `linear_regression_dataset.csv`  
**Rows:** 500 patient records  
**Features:**

| Column   | Description                        | Type    |
|----------|------------------------------------|---------|
| `AGE`    | Patient age (0–17)                 | int     |
| `FEMALE` | Gender indicator (1 = female)      | int     |
| `LOS`    | Length of hospital stay (days)     | int     |
| `RACE`   | Race category                      | float   |
| `APRDRG` | All Patient Refined DRG code       | int     |
| `TOTCHG` | **Target** — Total hospital charge | int     |

One missing value in `RACE` is imputed with the column mean.

## Pipeline

### 1. Data Loading & Exploration
- Load CSV with pandas
- Inspect shape, dtypes, null counts, and summary statistics

### 2. Preprocessing
- Fill missing values with column means
- Separate features (`X`) and target (`y`)
- Shuffle data with a fixed random seed (`np.random.seed(42)`) for reproducibility

### 3. Train/Test Split
- 80/20 split done manually using index slicing (no sklearn `train_test_split`)

### 4. Feature Scaling
- Z-score normalization using **training set statistics only** (mean and std)
- Applied to both train and test sets to prevent data leakage
- Zero-std columns are protected from division by zero

### 5. Model — Gradient Descent Linear Regression

All core logic is implemented from scratch:

```python
predict(X, w, b)          # ŷ = Xw + b
MSE(y_pred, y_true)       # Mean Squared Error loss
compute_gradients(...)    # dw, db via analytical gradient
train(...)                # Full gradient descent loop
```

**Training config:**
- Learning rate: `0.01`
- Max iterations: `200,000`
- Convergence criterion: weight/bias update norm `< 1e-6`
- Converges at ~iteration 1069

### 6. Training Output (sample)

```
epoch0:    cost = 22,393,145.87
epoch100:  cost =  8,276,864.30
epoch500:  cost =  7,976,928.12
Converged at iteration 1069
```

## Requirements

```
numpy
pandas
matplotlib
```