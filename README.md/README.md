# Financial Sentiment Classification using SEC 10-K Filings

## Overview

This project builds a machine learning pipeline to classify the sentiment of SEC 10-K filing text into:

* Positive
* Neutral
* Negative

The system uses Natural Language Processing (NLP) techniques and ensemble machine learning models to analyze financial reports and predict sentiment from company disclosures.

---

## Dataset

Dataset Source:

* winterForestStump/10-K_sec_filings (Hugging Face)

Sections Used:

* Business
* Management Discussion & Analysis (MD&A)
* Financial Statements

---

## Preprocessing

The following preprocessing steps were applied:

1. Convert text to lowercase
2. Remove numbers
3. Remove special characters
4. Normalize whitespace
5. Combine Business and MD&A sections
6. Limit text length to 3000 words

---

## Feature Engineering

TF-IDF Vectorization was used to convert text into numerical features.

Parameters:

* max_features = 3000
* stop_words = "english"
* min_df = 3
* max_df = 0.85
* ngram_range = (1,2)
* sublinear_tf = True

---

## Models Trained

The following models were trained and compared:

1. XGBoost
2. AdaBoost
3. CatBoost

---

## Evaluation

Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Best Model:

* XGBoost

Results:

* Accuracy: 78.63%
* F1 Score: 76.55%

---

## API Deployment

FastAPI was used to deploy the trained model.

Available Endpoints:

### GET /

Health check endpoint.

### GET /classes

Returns available prediction classes.

### POST /predict

Predicts sentiment for input text.

Example Request:

{
"text": "The company achieved record revenue growth and strong earnings."
}

Example Response:

{
"label": "positive",
"confidence": 0.74
}

---

## Project Structure

financial_sentiment_classifier/

├── api/

│ └── app.py

├── src/

│ ├── preprocess.py

│ ├── features.py

│ ├── train.py

│ ├── evaluate.py

│ └── utils.py

├── models/

├── reports/

├── data/

├── requirements.txt

└── README.md

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* CatBoost
* FastAPI
* Joblib
* Hugging Face Datasets

---

## Future Improvements

* Better sentiment labeling strategy
* Larger dataset
* Deep learning models (BERT/FinBERT)
* Improved class balancing
* Enhanced deployment and monitoring
