# Human Activity Recognition from Wearable Sensor Time Series

A classical machine learning pipeline that classifies six physical activities — walking, walking upstairs, walking downstairs, sitting, standing, and laying — from smartphone accelerometer and gyroscope time-series data. The project covers full exploratory data analysis, PCA-based dimensionality reduction and outlier detection, and a head-to-head benchmark of six classifiers.

## Overview

Wearable and smartphone-based activity recognition underpins fitness tracking, fall detection, and context-aware mobile applications. This project uses the **UCI Human Activity Recognition (HAR) with Smartphones** dataset — pre-extracted time and frequency domain features from accelerometer and gyroscope signals — to benchmark how well standard ML classifiers separate six everyday activities, and how much of that performance can be preserved after aggressive dimensionality reduction.

## Dataset

- **Source:** [UCI HAR Dataset (Kaggle mirror)](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones/data)
- **Size:** 10,299 samples × 563 features (561 sensor-derived features + `subject` ID + `Activity` label)
- **Classes (6):** `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
- **Quality:** No duplicate rows, no null values
- Each row is a pre-engineered feature vector (mean, std, mad, energy, correlation, frequency-domain features, etc.) computed from a windowed accelerometer/gyroscope signal — not raw time series.

## Exploratory Data Analysis

The notebook walks through a full EDA pass before any modeling:

- **Class balance** — count plot and percentile pie chart of activity distribution across the dataset.
- **3D PCA visualization** — projects all 561 features into 3 principal components and plots activities in 3D (via Plotly) to visually inspect class separability. `LAYING` in particular separates cleanly from the dynamic activities.
- **Correlation heatmap** — top 10 features most correlated with the activity label.
- **Label encoding** — activities mapped to integers: `{'LAYING': 0, 'SITTING': 1, 'STANDING': 2, 'WALKING': 3, 'WALKING_DOWNSTAIRS': 4, 'WALKING_UPSTAIRS': 5}`.
- **Outlier detection** — PCA-reduced data (29 components) scored with z-scores; points with any component beyond ±3 std flagged as outliers (1,173 of 10,299 rows, ~11%), visualized in a 2D PCA scatter plot.

## Feature Engineering

- **Optimal component selection:** cumulative explained-variance analysis shows **29 principal components retain 95% of the variance** in the original 561-dimensional feature space — a >19x reduction in dimensionality.
- This 29-component PCA representation is used specifically for the SVM-PCA benchmark model (see below), to test how much performance survives aggressive compression.

## Models Benchmarked

Six classifiers were trained on an 80/20 train/test split (full 561-feature space, unless noted):

| Model | Configuration |
|---|---|
| **SVM** | RBF kernel, `C=100`, full feature set |
| **SVM + PCA** | RBF kernel, `C=100`, trained on 29 PCA components (70/15/15 train/val/test split) |
| **Decision Tree** | entropy criterion, `max_depth=8` |
| **Logistic Regression** | `max_iter=1000` |
| **Random Forest** | 100 estimators, `min_samples_leaf=10`, `bootstrap=False` |
| **XGBoost** | multi-class softmax objective, 6 classes |

## Results

| Model | Test Accuracy | Precision (macro) | Recall (macro) | **F1-score (macro)** |
|---|---|---|---|---|
| **SVM (full features)** | **99.32%** | 0.9937 | 0.9936 | **0.9936** |
| XGBoost | 99.13% | 0.9912 | 0.9912 | 0.9912 |
| SVM + PCA (29 components) | 98.96% | 0.9896 | 0.9899 | 0.9898 |
| Logistic Regression | 98.45% | 0.9848 | 0.9850 | 0.9849 |
| Random Forest | 97.38% | 0.9732 | 0.9733 | 0.9732 |
| Decision Tree | 94.17% | 0.9401 | 0.9390 | 0.9395 |

**Key findings:**
- **SVM with the RBF kernel on the full feature set was the top performer**, achieving 99.32% test accuracy and a 0.9936 macro F1-score across all six activity classes.
- Compressing to **29 PCA components (a 19x reduction) cost only ~0.4 points of F1-score** (0.9936 → 0.9898) versus the full-feature SVM — a strong result for use cases where lower dimensionality matters (e.g. real-time or on-device inference).
- **XGBoost was the closest full-feature competitor** to SVM (99.13% accuracy) and reached perfect training accuracy (100%), indicating some overfitting relative to its test performance, though generalization was still strong.
- The **Decision Tree was the weakest model** (94.17% accuracy), as expected — single trees generalize less well than ensembles or margin-based methods on high-dimensional feature sets like this one.
- Per-model normalized confusion matrices (see notebook Section 4.3) show that most misclassification, across all models, occurs between the **SITTING and STANDING** classes — the two static postures with the most similar sensor signatures — while dynamic activities (walking variants) and `LAYING` are separated almost perfectly.

## Tech Stack

- **ML:** scikit-learn (SVM, Decision Tree, Logistic Regression, Random Forest, PCA), XGBoost
- **Data Handling:** Pandas, NumPy, SciPy (`zscore` for outlier detection)
- **Visualization:** Matplotlib, Seaborn, Plotly Express (3D PCA scatter)
- **Environment:** Google Colab, Google Drive for dataset storage

## Project Structure (Notebook Sections)

```
1. Data Collection and Preprocessing
   1.1  Libraries
   1.2  Dataset import (from Google Drive)
   1.3  Dataset description (shape, describe())
   1.4  Duplicate / null value check
   1.5  Class imbalance check
   1.6  Activity distribution (percentile pie chart)
   1.7  3D PCA visualization of activities
   1.8  Label encoding
   1.9  Correlation heatmap of top 10 features
   1.10 Outlier detection with PCA + z-score
   1.11 2D visualization of outliers

2. Feature Engineering and Selection
   2.1  Optimal PCA component count (95% variance threshold)
   2.2  Dimensionality reduction with PCA

3. Model Selection and Training
   3.1  Train/test split
   3.2  SVM
   3.3  SVM + PCA
   3.4  Decision Tree
   3.5  Logistic Regression
   3.6  Random Forest
   3.7  XGBoost

4. Model Evaluation and Performance
   4.1  Precision / recall / F1 for all models
   4.2  Metrics for SVM-PCA model
   4.3  Confusion matrices (normalized) for all models
   4.4  F1-score comparison bar chart across all models
```

## Setup & Usage

This notebook was built and run in **Google Colab** with the dataset stored on Google Drive.

1. Open the notebook in Google Colab (badge above), or run locally with Jupyter.
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones/data) or the linked [Google Drive folder](https://drive.google.com/drive/folders/19LS_bEokRTWYnxLlL3-0iwnuixFkZIP8?usp=drive_link).
3. Update the CSV path in Section 1.2 to point to your local/Drive copy of `harDataset.csv`.
4. Run cells sequentially — Sections 1 and 2 (EDA and PCA) feed into Section 3 (model training), and Section 4 requires the trained models from Section 3 to already be in memory.

### Requirements

```
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
seaborn
plotly
```

## Data Access

The UCI HAR Dataset is publicly available:
- [Kaggle mirror](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones/data)
- [Original UCI Machine Learning Repository page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

The raw CSV is not redistributed in this repository — download it separately and update the path in Section 1.2.

## Limitations & Future Work

- Features are pre-extracted (time/frequency-domain summary statistics) rather than raw accelerometer/gyroscope streams, so this pipeline doesn't test end-to-end deep learning on raw time series (e.g. 1D-CNN or LSTM approaches) — a natural extension for comparison.
- SITTING/STANDING confusion across all models suggests additional features (e.g. barometric or orientation-specific signals) or a hierarchical classifier (static vs. dynamic activity first, then subclass) could push accuracy further on the hardest class pair.
- No hyperparameter search (e.g. grid/random search, cross-validation) was performed — reported results reflect a single fixed configuration per model.

## Author

**Md Abrar Karim**
[GitHub](https://github.com/AbrarKarim01) · [LinkedIn](http://www.linkedin.com/in/karim-abrar) · [Portfolio](https://abrarkarim01.github.io/AbrarKarim.github.io/)
