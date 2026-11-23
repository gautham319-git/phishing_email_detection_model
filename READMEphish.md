# Phishing Email Detection Model

A machine learning model built with Scikit-learn that classifies emails as **Phishing** or **Safe** using both the textual content and URL features of each email.

## Features

- Trains on a labelled dataset of phishing and legitimate emails (24 samples, balanced 12/12)
- Extracts and analyses email features:
  - **Text:** TF-IDF representation of the email body
  - **URL features:** number of URLs, IP-address URLs, link shorteners (bit.ly, tinyurl), `@` symbol in URLs, hyphens in URLs
  - **Keywords:** count of suspicious words (urgent, verify, password, login, etc.)
  - **Style:** number of exclamation marks
- Classifies emails as "Phishing" or "Safe"
- Displays accuracy, classification report and confusion matrix

## How it works

1. Build a labelled dataset (1 = Phishing, 0 = Safe).
2. Extract hand-crafted URL and keyword features from each email.
3. Convert the text to TF-IDF vectors and join them with the extracted features.
4. Split the data 75/25 (stratified) into training and test sets.
5. Train a Logistic Regression classifier.
6. Evaluate on the test set and plot the confusion matrix.

## Requirements

```
pip install pandas scikit-learn scipy matplotlib
```

## Usage

```
python p_email.py
```

The script prints the metrics, shows the confusion matrix, saves it as `confusion_matrix.png`, and classifies two sample emails.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.83 |
| Phishing precision / recall | 1.00 / 0.67 |
| Safe precision / recall | 0.75 / 1.00 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted Safe | Predicted Phishing |
|---|---|---|
| **Actual Safe** | 3 | 0 |
| **Actual Phishing** | 1 | 2 |

Sample predictions:

- `Urgent! Verify your password at http://bit.ly/abc123` → **Phishing**
- `Can we move our meeting to 4 PM?` → **Safe**

## Limitations and future work

- The dataset is very small, so the test set has only 6 emails and the accuracy can change noticeably with a different split.
- For better results, train on a larger public dataset such as the Kaggle Phishing Email Dataset or the SpamAssassin corpus.
- Possible improvements: more URL features (domain age, HTTPS check), header analysis, and trying other models (Random Forest, SVM).
