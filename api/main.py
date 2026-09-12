#!/usr/bin/env python
"""
FastAPI inference server for the fake/real news classifier.

Loads the model + vectorizer trained by train_model.py (repo_root/model/)
and exposes a single /predict endpoint the browser extension calls with a
page's title/text.

Run from the repo root after training a model:
    uvicorn api.main:app --reload
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "lerabyte_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "lerabyte_vectorizer.joblib"

if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
    raise RuntimeError(
        f"No trained model found at {MODEL_PATH}.\n"
        "Run `python train_model.py` from the repo root first."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
feature_names = vectorizer.get_feature_names_out()

app = FastAPI(title="fact-check-ml API")

# Dev-only: allow any origin so the unpacked extension (chrome-extension://<id>)
# can call this during local development. Lock this down to the extension's
# real origin before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

TOP_N_TERMS = 8


class PredictRequest(BaseModel):
    title: str = ""
    text: str = ""


class TopTerm(BaseModel):
    term: str
    weight: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top_terms: list[TopTerm]
    # "coefficient": weight is a signed contribution to this prediction
    # (positive leans REAL, negative leans FAKE) -- only possible for
    # linear models. "salience": the model has no per-feature weights
    # (e.g. Random Forest), so this instead lists the highest-TF-IDF
    # terms present in the text -- notable words, not a causal explanation.
    explanation_type: str


def top_terms_for(vec):
    row = vec.tocoo()
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        pairs = [(feature_names[j], row.data[i] * coef[j]) for i, j in enumerate(row.col)]
        pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return pairs[:TOP_N_TERMS], "coefficient"

    pairs = [(feature_names[j], row.data[i]) for i, j in enumerate(row.col)]
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return pairs[:TOP_N_TERMS], "salience"


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    content = f"{req.title.strip()}. {req.text.strip()}".strip(". ").strip()
    if not content:
        raise HTTPException(status_code=400, detail="title and text are both empty")

    vec = vectorizer.transform([content])
    pred = model.predict(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(vec)[0][pred] * 100)
    else:
        # decision_function isn't a probability -- squash its magnitude
        # into a rough 0-100 display value.
        score = abs(float(model.decision_function(vec)[0]))
        confidence = float(min(99.0, 50 + score * 25))

    pairs, explanation_type = top_terms_for(vec)
    top_terms = [TopTerm(term=term, weight=round(float(w), 4)) for term, w in pairs]

    return PredictResponse(
        label=label,
        confidence=round(confidence, 2),
        top_terms=top_terms,
        explanation_type=explanation_type,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
