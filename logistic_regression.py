
# ДЗ 7. Logistic regression

import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# LOAD DATA

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

targets = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"]  = test["comment_text"].fillna("")

# EDA Basic

train[targets].mean().sort_values(ascending=False)
train["len"] = train["comment_text"].str.len()
train["words"] = train["comment_text"].str.split().map(len)
train[["len","words"]].describe()

# Підготовка даних + TF-IDF

X_text = train["comment_text"].values
X_test_text = test["comment_text"].values
Y = train[targets].values

X_tr, X_va, y_tr, y_va = train_test_split(
    X_text, Y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="word",
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1,2),
    min_df=2,
    max_features=200000,
    sublinear_tf=True,
    smooth_idf=True,
    norm="l2",
)

# Моделінг: 6 логрег (окремо під кожен таргет)

X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_va_tfidf = vectorizer.transform(X_va)

models = {}
va_pred = np.zeros((len(X_va), len(targets)))

for j, col in enumerate(targets):
    clf = LogisticRegression(
        C=4.0,
        solver="liblinear",   # добре для sparse
        max_iter=1000
    )
    clf.fit(X_tr_tfidf, y_tr[:, j])
    va_pred[:, j] = clf.predict_proba(X_va_tfidf)[:, 1]
    models[col] = clf

# Валідація: column-wise ROC AUC + mean (Kaggle)

aucs = []
for j, col in enumerate(targets):
    auc = roc_auc_score(y_va[:, j], va_pred[:, j])
    aucs.append(auc)
    print(f"{col:15s} ROC AUC = {auc:.5f}")

print("-"*40)
print(f"Mean column-wise ROC AUC = {np.mean(aucs):.5f}")

# A) Забираємо параметри з навченого TF-IDF та моделі

chosen = "toxic"
clf = models[chosen]

vocab = vectorizer.vocabulary_
idf = vectorizer.idf_.astype(np.float64)
token_pattern = re.compile(vectorizer.token_pattern)

w = clf.coef_.ravel().astype(np.float64)
b = float(clf.intercept_[0])
V = len(vocab)

# “CSR” спарс-матриця для зберігання tf-idf та множення на вектор ваг

class CSRMatrix:
    def __init__(self, indptr, indices, data, shape):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape

    def dot_dense_vector(self, w):
        # returns z = X @ w
        n_rows = self.shape[0]
        out = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            s = 0.0
            start, end = self.indptr[i], self.indptr[i+1]
            idx = self.indices[start:end]
            dat = self.data[start:end]
            for k in range(len(idx)):
                s += dat[k] * w[idx[k]]
            out[i] = s
        return out
    
# своя векторизация (tokenize + ngrams + tf-idf + l2 norm

def tokenize_words(text: str):
    text = text.lower()
    return token_pattern.findall(text)

def generate_ngrams(tokens, ngram_range=(1,2)):
    n1, n2 = ngram_range
    out = []
    L = len(tokens)
    for n in range(n1, n2+1):
        if n == 1:
            out.extend(tokens)
        else:
            for i in range(L - n + 1):
                out.append(" ".join(tokens[i:i+n]))
    return out

def texts_to_tfidf_csr(texts, vocab, idf, ngram_range=(1,2), sublinear_tf=True):
    indptr = [0]
    indices = []
    data = []

    for text in texts:
        tokens = tokenize_words(text)
        feats = generate_ngrams(tokens, ngram_range=ngram_range)

        counts = {}
        for f in feats:
            j = vocab.get(f)
            if j is None:
                continue
            counts[j] = counts.get(j, 0) + 1

        if not counts:
            indptr.append(indptr[-1])
            continue

        row_idx = np.fromiter(counts.keys(), dtype=np.int64)
        row_tf  = np.fromiter(counts.values(), dtype=np.float64)

        if sublinear_tf:
            row_tf = 1.0 + np.log(row_tf)

        row_val = row_tf * idf[row_idx]

        # l2 norm
        norm = np.sqrt(np.sum(row_val * row_val))
        if norm > 0:
            row_val = row_val / norm

        # сортуємо індекси як робить sklearn
        order = np.argsort(row_idx)
        row_idx = row_idx[order]
        row_val = row_val[order]

        indices.extend(row_idx.tolist())
        data.extend(row_val.tolist())
        indptr.append(len(indices))

    return CSRMatrix(
        indptr=np.array(indptr, dtype=np.int64),
        indices=np.array(indices, dtype=np.int32),
        data=np.array(data, dtype=np.float64),
        shape=(len(texts), len(vocab))
    )

# Ручний інференс: sigmoid(w·x + b)

def sigmoid(z):
    # стабільний sigmoid
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

# 1) Порівняння scratch vs sklearn на валідації
sample_texts_val = X_va[:2000]

# sklearn pipeline
X_sklearn = vectorizer.transform(sample_texts_val)
p_sklearn = clf.predict_proba(X_sklearn)[:, 1]

# from scratch pipeline
t0 = time.time()
X_my = texts_to_tfidf_csr(
    sample_texts_val, vocab=vocab, idf=idf,
    ngram_range=vectorizer.ngram_range,
    sublinear_tf=vectorizer.sublinear_tf
)
z_my = X_my.dot_dense_vector(w) + b
p_my = sigmoid(z_my)
t1 = time.time()

diff = np.abs(p_sklearn - p_my)
print("Max abs diff:", diff.max())
print("Mean abs diff:", diff.mean())
print("Median abs diff:", np.median(diff))

# sanity check: кореляція
corr = np.corrcoef(p_sklearn, p_my)[0,1]
print("Correlation:", corr)

y_true = y_va[:2000, targets.index(chosen)]
auc_my = roc_auc_score(y_true, p_my)
auc_sk = roc_auc_score(y_true, p_sklearn)
print("AUC scratch:", auc_my)
print("AUC sklearn:", auc_sk)

test_texts = X_test_text
t2 = time.time()
X_test_my = texts_to_tfidf_csr(
    test_texts, vocab=vocab, idf=idf,
    ngram_range=vectorizer.ngram_range,
    sublinear_tf=vectorizer.sublinear_tf
)
z_test_my = X_test_my.dot_dense_vector(w) + b
p_test_my = sigmoid(z_test_my)
t3 = time.time()

X_test_sk = vectorizer.transform(test_texts)
p_test_sk = clf.predict_proba(X_test_sk)[:, 1]

# Порівняння результатів

test_diff = np.abs(p_test_sk - p_test_my)
print("Test Max abs diff:", test_diff.max())
print("Test Mean abs diff:", test_diff.mean())
print("Validation scratch pipeline time (sec):", round(t1 - t0, 2))
print("Test scratch pipeline time (sec):", round(t3 - t2, 2))

if (t3 - t2) > 120:
    print("УВАГА: from-scratch pipeline займає багато часу на повному test.csv")


def build_all_plots(
    train, targets,
    y_va, va_pred,
    vectorizer, models,
    chosen="toxic",
    p_sklearn=None, p_scratch=None
):
    # ---------- 1) Pos rate ----------
    pos_rate = train[targets].mean().sort_values(ascending=False)
    plt.figure(figsize=(7, 4), dpi=120)
    plt.bar(pos_rate.index, pos_rate.values)
    plt.title("Positive rate by class")
    plt.xlabel("Class")
    plt.ylabel("Positive rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # ---------- 2) Comment length histogram ----------
    lens = train["comment_text"].fillna("").str.len().to_numpy()
    lens = np.clip(lens, 0, 5000)
    plt.figure(figsize=(7, 4), dpi=120)
    plt.hist(lens, bins=40)
    plt.title("Comment length distribution (clipped to 5000 chars)")
    plt.xlabel("Length (chars)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # ---------- 3) ROC curves (6 classes) ----------
    plt.figure(figsize=(7, 4), dpi=120)
    for j, col in enumerate(targets):
        y_true_local = y_va[:, j]
        y_score = va_pred[:, j]
        fpr, tpr, _ = roc_curve(y_true_local, y_score)
        auc_local = roc_auc_score(y_true_local, y_score)
        plt.plot(fpr, tpr, label=f"{col} (AUC={auc_local:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title("ROC curves (validation)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # ---------- 4) AUC bars + mean line ----------
    aucs_local = [roc_auc_score(y_va[:, j], va_pred[:, j]) for j in range(len(targets))]
    mean_auc = float(np.mean(aucs_local))
    plt.figure(figsize=(7, 4), dpi=120)
    plt.bar(targets, aucs_local)
    plt.axhline(mean_auc, linestyle="--", linewidth=1, label=f"Mean AUC={mean_auc:.3f}")
    plt.title("ROC AUC by class (validation)")
    plt.xlabel("Class")
    plt.ylabel("ROC AUC")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- 5) Top tokens by weight (chosen class) ----------
    clf_local = models[chosen]
    w_local = clf_local.coef_.ravel()
    feature_names = np.array(vectorizer.get_feature_names_out())

    topn = 20
    idx_pos = np.argsort(w_local)[-topn:][::-1]
    idx_neg = np.argsort(w_local)[:topn]

    tokens = np.concatenate([feature_names[idx_neg], feature_names[idx_pos]])
    weights = np.concatenate([w_local[idx_neg], w_local[idx_pos]])

    order = np.argsort(weights)
    tokens = tokens[order]
    weights = weights[order]

    plt.figure(figsize=(7, 5), dpi=120)
    plt.barh(tokens, weights)
    plt.title(f"Top token weights (chosen='{chosen}')")
    plt.xlabel("Weight (coef)")
    plt.tight_layout()
    plt.show()

    # ---------- 6) Prediction distribution (chosen: pos vs neg) ----------
    j = targets.index(chosen)
    y_true_local = y_va[:, j]
    y_score = va_pred[:, j]

    plt.figure(figsize=(7, 4), dpi=120)
    plt.hist(y_score[y_true_local == 0], bins=40, alpha=0.7, label="y=0")
    plt.hist(y_score[y_true_local == 1], bins=40, alpha=0.7, label="y=1")
    plt.title(f"Predicted probability distribution (chosen='{chosen}')")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- 7) Scratch vs sklearn scatter ----------
    if p_sklearn is not None and p_scratch is not None:
        p_sklearn = np.asarray(p_sklearn)
        p_scratch = np.asarray(p_scratch)

        plt.figure(figsize=(6, 6), dpi=120)
        plt.scatter(p_sklearn, p_scratch, s=8, alpha=0.5)
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        diff_local = np.abs(p_sklearn - p_scratch)
        corr_local = np.corrcoef(p_sklearn, p_scratch)[0, 1]
        plt.title(f"Sklearn vs Scratch (max diff={diff_local.max():.2e}, corr={corr_local:.6f})")
        plt.xlabel("Sklearn probability")
        plt.ylabel("Scratch probability")
        plt.tight_layout()
        plt.show()
    else:
        print("Scatter пропущен: передай p_sklearn и p_scratch (p_my).")


build_all_plots(
    train=train,
    targets=targets,
    y_va=y_va,
    va_pred=va_pred,
    vectorizer=vectorizer,
    models=models,
    chosen=chosen,
    p_sklearn=p_sklearn,
    p_scratch=p_my
)

    