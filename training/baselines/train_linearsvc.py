"""
LinearSVC + char n-gram TF-IDF baseline.

This was the project's original classifier before the BiLSTM-Attention model,
kept as the classical baseline in the comparison.

No GPU needed. Runs in ~1 minute.
"""

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from data_utils import (
    LABEL_TO_IDX,
    LABELS,
    build_text_input,
    load_split,
)
from evaluate_common import save_results


def df_to_texts_and_labels(df):
    texts = []
    labels = []
    for _, row in df.iterrows():
        method = str(row.get("method", "GET"))
        path = str(row.get("path", "/"))
        query = str(row.get("query_string", ""))
        body = str(row.get("body", ""))
        if query == "nan":
            query = ""
        if body == "nan":
            body = ""
        texts.append(build_text_input(method, path, query, body))
        labels.append(LABEL_TO_IDX[row["label"]])
    return texts, np.array(labels, dtype=np.int64)


def main():
    train_df = load_split("train")
    test_df = load_split("test")
    print(f"train={len(train_df)}  test={len(test_df)}")

    train_texts, y_train = df_to_texts_and_labels(train_df)
    test_texts, y_test = df_to_texts_and_labels(test_df)

    # Char n-gram TF-IDF (3-5) — captures obfuscation patterns at sub-word level
    print("Fitting TF-IDF (char n-grams 3-5)...")
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
    )
    t0 = time.perf_counter()
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    t_vec = time.perf_counter() - t0
    print(f"TF-IDF fit/transform: {t_vec:.1f}s")
    print(f"Feature dim: {X_train.shape[1]}")

    # LinearSVC with class balancing
    print("Training LinearSVC...")
    t0 = time.perf_counter()
    clf = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000, dual="auto")
    clf.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    print(f"Train time: {t_train:.1f}s")

    # Predict
    print("Predicting on test set...")
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    t_pred = time.perf_counter() - t0

    # Approximate per-sample latency
    n_test = len(y_test)
    avg_ms = (t_pred / n_test) * 1000.0
    # crude p99 estimate via single-sample timing
    print("Measuring per-sample latency (n=2000)...")
    sample_idx = np.random.RandomState(42).choice(n_test, size=min(2000, n_test), replace=False)
    lats = []
    for i in sample_idx:
        t0 = time.perf_counter()
        _ = clf.predict(X_test[i])
        lats.append((time.perf_counter() - t0) * 1000.0)
    lats = np.array(lats)
    latency = {
        "p50": float(np.percentile(lats, 50)),
        "p95": float(np.percentile(lats, 95)),
        "p99": float(np.percentile(lats, 99)),
        "mean": float(lats.mean()),
        "n": len(lats),
        "note": "single-sample sklearn predict; includes Python overhead",
    }
    print(f"Latency p99: {latency['p99']:.2f} ms")

    # Approximate "params" — non-zero TF-IDF features × num classes
    n_params = X_train.shape[1] * len(LABELS)

    save_results(
        model_name="linearsvc",
        y_true=y_test,
        y_pred=y_pred,
        latency=latency,
        n_params=n_params,
        training_time_sec=t_train + t_vec,
        epochs_trained=None,
        best_val_accuracy=None,
        extra={
            "tfidf_features": int(X_train.shape[1]),
            "tfidf_ngram_range": [3, 5],
            "tfidf_analyzer": "char_wb",
            "C": 1.0,
        },
    )


if __name__ == "__main__":
    main()
