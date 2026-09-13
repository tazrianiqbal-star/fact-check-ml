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

The popup grabs the page's title + visible article/paragraph text, sends it to the local API, and shows a REAL/FAKE badge, a confidence score, and the words that most influenced the prediction (signed toward REAL/FAKE for linear models; just the most notable present words for non-linear ones like Random Forest, labeled accordingly since that's not a causal explanation).

# Deploying the API for free
Dockerfile + render.yaml are set up for Render's free tier:

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. In Render, choose "New" -> "Blueprint", point it at this repo. It reads render.yaml automatically and creates a free web service named `fact-check-ml-api`.
3. The Docker build trains a fresh model as part of the build (train_model.py) rather than shipping a committed model file -- the trained Random Forest is ~175MB, over GitHub's 100MB plain-file limit, so baking it in at build time avoids needing Git LFS.
4. Once deployed, Render gives you a URL like `https://fact-check-ml-api.onrender.com`. Update `API_URL` in extension/popup.js to point at it (plus `/predict`), and add that origin to `host_permissions` in extension/manifest.json.

Two free-tier caveats worth knowing going in:
- **Cold starts**: free instances spin down after 15 minutes of inactivity; the first request after that can take 30+ seconds while it wakes back up.
- **512MB RAM limit**: the current Random Forest model may be tight against this once loaded alongside scikit-learn/pandas. If the deployed service crashes or won't start, the fix is forcing train_model.py to pick a linear model (Logistic Regression or Linear SVC) instead -- a few hundred KB versus 175MB, and it also unlocks signed word-level explanations in the extension instead of the current salience-only view.

The API's CORS policy (`allow_origins=["*"]`) is intentionally left open rather than dev-only, since this endpoint doesn't handle auth or sensitive data and browser extensions typically call from unpredictable origins.

# Publishing the extension
The extension is already cross-browser: it uses the `browser.*` WebExtension namespace everywhere (via Mozilla's `webextension-polyfill`, vendored at extension/vendor/browser-polyfill.min.js) instead of Chrome-only `chrome.*`, and manifest.json declares both `background.service_worker` (Chrome) and `background.scripts` (Firefox, which doesn't run MV3 background code as a service worker) so the same package loads in both. It also sets the `browser_specific_settings.gecko.id` and `data_collection_permissions` keys Firefox requires for MV3 submissions -- the latter is set to `websiteContent`, since the extension does send the current page's title/text to the API for classification.

Chrome, Firefox, and Edge each still require their own store submission:
- **Chrome Web Store**: one-time $5 developer registration fee, then submit via the [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole). Review typically takes a few days.
- **Firefox Add-ons (AMO)**: free, submit via [addons.mozilla.org/developers](https://addons.mozilla.org/developers/). Worth testing in real Firefox before submitting -- this was built and reasoned through carefully, but not click-tested in an actual Firefox install.
- **Edge Add-ons**: free, via the Microsoft Partner Center. Since Edge is Chromium-based, the existing Manifest V3 package works with little to no change.

# Limitations
Originally trained on a single, dated dataset (2016-era US political news), which generalized poorly -- ordinary out-of-domain headlines scored only ~62% on the out-of-domain eval set (see evaluate_ood.py) despite ~93% in-domain test accuracy. Adding GossipCop and COVID-19 misinformation data brought out-of-domain accuracy up to ~94% on that same eval set, at the cost of some in-domain accuracy (~93% -> ~85%). The out-of-domain eval set is hand-written, not an independently verified benchmark -- treat its numbers as a useful signal, not a rigorous one. TF-IDF + linear/tree models also have a ceiling: they learn lexical/stylistic patterns, not whether a claim is factually true, so a confident, well-written fabrication can still fool them.

