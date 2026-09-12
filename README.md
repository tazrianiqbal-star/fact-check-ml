# fact-check-ml
# Fake news detector
This web app functions as an extension to distinguish between real vs fake news, in an attempt to reduce noise and misinformation in the academic data collection processes. This is my first time building an ML model from scratch ready for deployment, hope it comes of use to someone! Dataset: www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset 


# Setup
bash
pip install -r requirements.txt
# Usage
bash
python train_model.py


This automatically downloads three datasets (unless already cached locally) spanning different topics/eras -- 2016 US political news, entertainment/celebrity news (GossipCop), and 2020 COVID-19 health misinformation -- compares four models (Logistic Regression, Naive Bayes, Linear SVC, Random Forest) via 5-fold cross-validation, trains the best one, prints evaluation metrics, and saves the model + vectorizer to model/.
Use the already-trained model (no retraining needed) by loading the files in model/ with joblib.load().

# Evaluating generalization
bash
python evaluate_ood.py


train_model.py's held-out test accuracy is measured on a split of the same data it trained on, so it mostly measures whether the model recognizes more of that data's style, not whether it generalizes. evaluate_ood.py instead scores the trained model against eval/ood_examples.csv, a small hand-written set of headlines spanning topics/eras/styles the training data doesn't cover -- a more honest signal of real-world performance. Regenerate/extend that set by editing eval/build_ood_examples.py and rerunning it.

# Browser extension
A Manifest V3 Chrome extension (extension/) and a small FastAPI server (api/) that serves the trained model, so the check can run against whatever page is open in the browser.

Run the API (after training a model, since it loads model/ at startup):
bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload


Then load the extension:
1. Go to chrome://extensions, enable "Developer mode" (top right).
2. Click "Load unpacked" and select the extension/ folder.
3. Open any article page, click the extension icon, and click "Check this page".

The popup grabs the page's title + visible article/paragraph text, sends it to the local API, and shows a REAL/FAKE badge, a confidence score, and the words that most influenced the prediction (signed toward REAL/FAKE for linear models; just the most notable present words for non-linear ones like Random Forest, labeled accordingly since that's not a causal explanation). The API's CORS policy (`allow_origins=["*"]`) is dev-only -- restrict it to the extension's real origin before any public deployment, and swap `API_URL` in extension/popup.js from localhost to a deployed URL.

# Limitations
Originally trained on a single, dated dataset (2016-era US political news), which generalized poorly -- ordinary out-of-domain headlines scored only ~62% on the out-of-domain eval set (see evaluate_ood.py) despite ~93% in-domain test accuracy. Adding GossipCop and COVID-19 misinformation data brought out-of-domain accuracy up to ~94% on that same eval set, at the cost of some in-domain accuracy (~93% -> ~85%). The out-of-domain eval set is hand-written, not an independently verified benchmark -- treat its numbers as a useful signal, not a rigorous one. TF-IDF + linear/tree models also have a ceiling: they learn lexical/stylistic patterns, not whether a claim is factually true, so a confident, well-written fabrication can still fool them.

