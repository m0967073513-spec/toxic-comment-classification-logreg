# Toxic Comment Classification using Logistic Regression and TF-IDF

This project implements a multilabel text classification system for toxic comment detection using Logistic Regression and TF-IDF vectorization, based on the Jigsaw Toxic Comment Classification Challenge dataset.

The model predicts six classes:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The implementation includes both a standard sklearn pipeline and a fully custom from-scratch inference pipeline using manual TF-IDF, sparse matrix (CSR), and sigmoid inference.

## Results

### Validation ROC AUC scores:

| Class         | ROC AUC |
|---------------|---------|
| toxic         | 0.97501 |
| severe_toxic  | 0.98123 |
| obscene       | 0.98643 |
| threat        | 0.98962 |
| insult        | 0.98029 |
| identity_hate | 0.97332 |

**Mean ROC AUC: 0.98098**

### From-scratch pipeline verification:

| Metric        | Value      |
|---------------|------------|
| Correlation   | 1.0        |
| AUC sklearn   | 0.983226   |
| AUC scratch   | 0.983226   |
| Max abs diff  | 4.44e-16   |

This confirms the correctness of the custom TF-IDF, sparse matrix, and inference implementation.

## Features

- TF-IDF vectorization with n-grams (1,2)
- Six independent Logistic Regression models
- Column-wise ROC AUC validation
- Custom CSR sparse matrix implementation
- Custom TF-IDF vectorization
- Manual logistic regression inference using sigmoid(w·x+b)
- Verification against sklearn implementation
- Visualization of model performance and feature importance

## Project Structure

```
.
├── logistic_regression.py
├── README.md
├── requirements.txt
└── plots/ (optional)
```

## Installation

Create virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download dataset from Kaggle:

https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Place files in project directory:

- `train.csv`
- `test.csv`

## Run

Execute:

```bash
python logistic_regression.py
```

The script will:

- train 6 Logistic Regression models
- evaluate ROC AUC
- run from-scratch TF-IDF and sparse inference
- compare results with sklearn
- generate performance plots

## From-Scratch Implementation

The project includes a manual implementation of:

- TF-IDF vectorization
- sparse CSR matrix
- logistic regression inference
- sigmoid probability calculation

Results match sklearn output with numerical precision.

## Author

Machine Learning project implemented as part of practical training and portfolio development.
