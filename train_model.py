#!/usr/bin/env python
"""
train_model.py

Fake vs. real news classifier.

Trains on the "Fake and Real News" dataset
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset),
or an automatically downloaded fallback with equivalent columns.
"""

import os
import ssl
import urllib.request

import certifi
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

FAKE_PATH = "Fake.csv"
TRUE_PATH = "True.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
FALLBACK_PATH = "fake_or_real_news.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "lerabyte_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "lerabyte_vectorizer.joblib")


def ensure_fake_true_csvs():
    if os.path.exists(FAKE_PATH) and os.path.exists(TRUE_PATH):
        return

    print(f"{FAKE_PATH} / {TRUE_PATH} not found -- downloading a public "
          f"equivalent dataset and splitting it to match...")
    if not os.path.exists(FALLBACK_PATH):
        # Use certifi's CA bundle explicitly, since some Python installs
        # don't ship a working default one.
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(FALLBACK_URL, context=ssl_context) as response:
            with open(FALLBACK_PATH, "wb") as out_file:
                out_file.write(response.read())

    combined = pd.read_csv(FALLBACK_PATH)
    fake = combined[combined["label"] == "FAKE"][["title", "text"]]
    true = combined[combined["label"] == "REAL"][["title", "text"]]
    fake.to_csv(FAKE_PATH, index=False)
    true.to_csv(TRUE_PATH, index=False)


ensure_fake_true_csvs()
fake = pd.read_csv(FAKE_PATH)
true = pd.read_csv(TRUE_PATH)

# label: 0 = fake news, 1 = real news
fake["label"] = 0
true["label"] = 1
data = pd.concat([fake, true], axis=0)
data = data[["title", "text", "label"]]

# combine title + text into one field the vectorizer can work with
data["content"] = data["title"].fillna("") + ". " + data["text"].fillna("")

# shuffle fake and real articles together
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# separate input data and target label
X = data["content"]
y = data["label"]

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# vectorize the text with TF-IDF
# ngram_range=(1, 2) keeps single words AND two-word phrases ("breaking
# news", "according to") -- bigrams often carry more signal than single
# words alone.
# max_features caps the vocabulary at the N most informative terms, which
# keeps training fast and avoids the model latching onto rare one-off words.
# min_df=3 drops any word/phrase that appears in fewer than 3 articles --
# those are usually typos or names too rare to generalize from.
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2),
    max_features=20_000,
    min_df=3,
)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# candidate models to compare:
#   - LogisticRegression: draws a linear boundary between "fake" and "real"
#   - MultinomialNB: probability-based classifier
#   - LinearSVC: strong on high-dimensional text
#   - RandomForestClassifier: decision trees, averaged
candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVC": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
print("Cross-validation results (5-fold, on training data only):\n")
cv_results = {}
for name, candidate in candidate_models.items():
    scores = cross_val_score(candidate, X_train_vectorized, y_train, cv=5, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name:<25} mean accuracy: {scores.mean() * 100:.2f} %   (+/- {scores.std() * 100:.2f} %)")

# pick the best-performing model by mean cross-validation accuracy, then
# train it on the FULL training set (cross-validation only used partial
# slices of it) and evaluate it on the held-out test set.
best_name = max(cv_results, key=lambda name: cv_results[name].mean())
print(f"\nBest model by cross-validation: {best_name}")

model = candidate_models[best_name]
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nHeld-out Test Accuracy: {accuracy * 100:.2f} %")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# save the trained pieces so predict_news() can be reused without retraining
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)


def predict_news(text: str):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label = "REAL news" if pred == 1 else "FAKE news"
    print(f"\nPrediction: {label}")

    # Not every model can report a confidence percentage the same way.
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0][pred]
        print(f"Confidence: {prob * 100:.2f} %")
    else:
        score = model.decision_function(vec)[0]
        print(f"Decision score: {score:.2f} (further from 0 = more confident)")


def predict_batch(articles):
    """
    Run the model on a list of articles and return the results as a table.

    `articles` can be either:
      - a list of strings, e.g. ["headline 1", "headline 2", ...]
      - a list of (text, expected_label) tuples, e.g.
        [("headline 1", "REAL"), ("headline 2", "FAKE"), ...]
        which also lets this report accuracy on your own examples.
    """
    rows = []
    for item in articles:
        # Figure out whether this item came with a known/expected answer.
        if isinstance(item, tuple):
            text, expected = item
        else:
            text, expected = item, None

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        predicted_label = "REAL" if pred == 1 else "FAKE"

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vec)[0][pred] * 100
        else:
            confidence = model.decision_function(vec)[0]

        row = {
            "text": text[:70] + ("..." if len(text) > 70 else ""),
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
        }
        if expected is not None:
            row["expected"] = expected
            row["correct"] = (predicted_label == expected)
        rows.append(row)

    results = pd.DataFrame(rows)

    if "correct" in results.columns:
        acc = results["correct"].mean() * 100
        print(f"Batch accuracy on your examples: {acc:.1f}% "
              f"({results['correct'].sum()}/{len(results)} correct)\n")

    return results


if __name__ == "__main__":
    predict_news("Scientists discover a new way to improve battery efficiency.")
    predict_news("Government confirms aliens built secret tunnels under every major city.")

    my_articles = [
        ("Local council approves new bike lane after months of public hearings.", "REAL"),
        ("Man claims he cured his cold by staring directly at the sun for an hour.", "FAKE"),
        ("Federal Reserve holds interest rates steady, citing inflation concerns.", "REAL"),
        ("Secret NASA memo reveals moon landing was filmed in a Nevada parking lot.", "FAKE"),
        ("City reports a 12% drop in water usage after new conservation program.", "REAL"),
    ]

    batch_results = predict_batch(my_articles)
    print(batch_results.to_string(index=False))
