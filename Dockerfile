FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt

# Train at build time rather than committing model artifacts: the trained
# Random Forest is ~175MB, over GitHub's 100MB plain-file limit, and this
# keeps the deployed model reproducible from the same pinned dependencies
# and datasets train_model.py already downloads automatically.
RUN python train_model.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
