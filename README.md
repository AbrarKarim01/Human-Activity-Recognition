# Human-Activity-Recognition

## Overview
Human Activity Recognition (HAR) using smartphone sensor signals. This project explores data preprocessing, feature engineering with PCA, and classification with multiple machine learning models. The goal is to classify six activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.

## Dataset
- Kaggle: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones/data
- Google Drive mirror: https://drive.google.com/drive/folders/19LS_bEokRTWYnxLlL3-0iwnuixFkZIP8?usp=drive_link

The notebook loads a CSV named `harDataset.csv` (update the path if you run locally).

## Workflow
1. Data collection and preprocessing
	- Check duplicates and nulls
	- Class imbalance visualization
	- Label encoding for activity labels
2. Feature engineering and selection
	- PCA visualization and variance analysis
	- Outlier detection using PCA
3. Model selection and training
	- SVM, SVM + PCA, Decision Tree, Logistic Regression, Random Forest, XGBoost
4. Evaluation
	- Precision, recall, F1-score (macro)
	- Confusion matrices
	- F1-score comparison

## Results
Macro F1-scores from the notebook:

| Model | F1-score |
| --- | --- |
| SVM | 0.9936 |
| Decision Tree | 0.9395 |
| Logistic Regression | 0.9849 |
| Random Forest | 0.9732 |
| XGBoost | 0.9912 |
| SVM + PCA | 0.9898 |

Best overall model: SVM achieved the highest macro F1-score.

## How to Run
1. Download the dataset and place `harDataset.csv` locally.
2. Open the notebook:
	- Human_Activity_Recognition.ipynb
3. Update the dataset path if running outside Colab.
4. Run all cells.

## Dependencies
- pandas, numpy
- seaborn, matplotlib, plotly
- scikit-learn
- xgboost
- scipy

Install example:
```bash
pip install pandas numpy seaborn matplotlib plotly scikit-learn xgboost scipy
```