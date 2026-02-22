# Toxic Comment Classification — Logistic Regression

A clean, production-ready Python project that classifies text comments as **toxic** or **non-toxic** using a TF-IDF feature extractor and a Logistic Regression classifier built with scikit-learn.

---

## Project Structure

```
toxic-comment-classification-logreg/
├── logistic_regression.py   # Full ML pipeline: data → features → model → evaluation
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignores data files, venvs, build artifacts
└── README.md
```

---

## Features

- **TF-IDF vectorisation** with unigrams and bigrams (up to 10 000 features)
- **Logistic Regression** with balanced class weights and L-BFGS solver
- Comprehensive evaluation: accuracy, ROC-AUC, and a full classification report
- Works out-of-the-box with a built-in demo dataset *or* your own CSV file
- Minimal dependencies, single-file entry point

---

## Quick Start

### 1 — Clone and install dependencies

```bash
git clone https://github.com/m0967073513-spec/toxic-comment-classification-logreg.git
cd toxic-comment-classification-logreg

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Run with built-in sample data

```bash
python logistic_regression.py
```

### 3 — Run with your own dataset

Provide a CSV file that contains at least two columns — `text` (string) and `toxic` (0 or 1):

```bash
python logistic_regression.py --data path/to/train.csv
```

---

## Example Output

```
No data file provided — using built-in sample data.
Dataset size: 20 rows | Toxic: 8 | Non-toxic: 12
Train: 16 | Test: 4

Training pipeline (TF-IDF + Logistic Regression) …
Training complete.

==================================================
Model Evaluation
==================================================
Accuracy : 0.7500
ROC-AUC  : 1.0000

Classification Report:
              precision    recall  f1-score   support

   non-toxic       1.00      0.50      0.67         2
       toxic       0.67      1.00      0.80         2

    accuracy                           0.75         4
   macro avg       0.83      0.75      0.73         4
weighted avg       0.83      0.75      0.73         4

==================================================
Example Predictions
==================================================
[TOXIC | 0.59] 'You are the most pathetic person alive.'
[SAFE | 0.45] 'I really appreciate your kind words, thank you!'
[TOXIC | 0.51] 'This post is garbage and so are you.'
[SAFE | 0.42] 'Wonderful tutorial, learned a lot!'
```

> **Note:** The above results are from the tiny 20-comment demo dataset.  
> On a real dataset (e.g. the Kaggle *Toxic Comment Classification Challenge*) expect ROC-AUC ≥ 0.96.

---

## Model Details

| Component | Choice | Rationale |
|---|---|---|
| Vectoriser | TF-IDF (1–2 grams, 10 k features, sublinear TF) | Captures word importance and common toxic phrases |
| Classifier | Logistic Regression (C=1, balanced weights, lbfgs) | Fast, interpretable, strong baseline for text |
| Split | 80 / 20 stratified | Preserves class ratio in both sets |

---

## Requirements

| Package | Minimum version |
|---|---|
| numpy | 1.24 |
| pandas | 2.0 |
| scikit-learn | 1.4 |

---

## License

[MIT](LICENSE)
