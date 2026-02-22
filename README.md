# Toxic Comment Classification (Jigsaw) — TF-IDF + Logistic Regression + From-Scratch Inference

Multilabel text classification for toxic comments based on the Kaggle “Jigsaw Toxic Comment Classification Challenge”.
The project trains 6 independent Logistic Regression models (one per label) using TF-IDF features and evaluates them with column-wise ROC AUC.
Additionally, it includes a from-scratch verification pipeline: custom TF-IDF, custom CSR sparse matrix, and manual inference with sigmoid(w·x+b),
validated against sklearn outputs.

## Labels
toxic, severe_toxic, obscene, threat, insult, identity_hate

## Dataset
Download `train.csv` and `test.csv` from Kaggle:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

Place the files in the project root (do not commit them to GitHub):
- train.csv
- test.csv

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## RUN
``` bash
python logistic_regression.py
```

Real Validation Results (ROC AUC, column-wise)
Output on real Kaggle dataset:

toxic           ROC AUC = 0.97501
severe_toxic    ROC AUC = 0.98123
obscene         ROC AUC = 0.98643
threat          ROC AUC = 0.98962
insult          ROC AUC = 0.98029
identity_hate   ROC AUC = 0.97332
----------------------------------------

Mean column-wise ROC AUC = 0.98098

From-Scratch Verification (chosen class = toxic)
The from-scratch pipeline reproduces sklearn’s TF-IDF + Logistic Regression inference with numerical precision:
---------------------------
Max abs diff: 4.440892098500626e-16
Mean abs diff: 5.3685422535428014e-18
Correlation: 1.0
AUC scratch: 0.983226016371545
AUC sklearn: 0.983226016371545
Validation scratch pipeline time: 0.21 sec
Test scratch pipeline time: 13.5 sec
------------------------------
Notes

train.csv / test.csv are intentionally excluded from Git tracking.
The script supports plotting (ROC curves, AUC bars, feature weights, probability distributions, sklearn vs scratch scatter) for better interpretability.

---

## License

[MIT](LICENSE)
