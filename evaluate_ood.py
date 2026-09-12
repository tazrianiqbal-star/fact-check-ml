#!/usr/bin/env python
"""
evaluate_ood.py

Scores the trained model against eval/ood_examples.csv -- a hand-written
set of articles spanning topics/eras the training data doesn't cover.

train_model.py's own held-out accuracy is measured on a split of the SAME
dataset it trained on, so a high score there mostly confirms the model
recognizes more of that dataset's style. This script measures something
different: does it generalize past that one dataset. Run it after every
retrain to see whether a change actually helped or just moved the
in-domain number.
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "lerabyte_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "lerabyte_vectorizer.joblib")
OOD_PATH = os.path.join("eval", "ood_examples.csv")


def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        raise SystemExit(
            f"No trained model found at {MODEL_PATH} / {VECTORIZER_PATH}.\n"
            "Run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    data = pd.read_csv(OOD_PATH)
    # Match the same content format train_model.py trains on.
    content = data["title"].fillna("") + ". " + data["text"].fillna("")
    y_true = (data["label"] == "REAL").astype(int)

    X = vectorizer.transform(content)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Out-of-domain accuracy: {accuracy * 100:.2f} %  ({len(data)} examples)\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    data["predicted"] = ["REAL" if p == 1 else "FAKE" for p in y_pred]
    data["correct"] = data["predicted"] == data["label"]
    misses = data[~data["correct"]][["title", "label", "predicted"]]
    if len(misses):
        print(f"\nMisclassified ({len(misses)}/{len(data)}):")
        print(misses.to_string(index=False))
    else:
        print("\nNo misclassifications.")


if __name__ == "__main__":
    main()
