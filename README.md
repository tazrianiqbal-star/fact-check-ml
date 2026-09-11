# fact-check-ml
# Fake news detector
This web app functions as an extension to distinguish between real vs fake news, in an attempt to reduce noise and misinformation in the academic data collection processes. This is my first time building an ML model from scratch ready for deployment, hope it comes of use to someone! Dataset: www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset 

This will automatically download a public dataset if Fake.csv/True.csv aren't present locally, compare four models (Logistic Regression, Naive Bayes, Linear SVC, Random Forest) via 5-fold cross-validation, train the best one, print evaluation metrics and save the model + vectorizer to model/.
Use the already-trained model (no retraining needed) by loading the files in model/ with joblib.load().
Limitations

Trained on a specific, dated dataset (2016-era US political news). Accuracy on topics/eras/styles outside that domain is meaningfully lower -- in informal testing, batches of ordinary, plausible headlines scored well below the model's ~90%+ reported test accuracy.

