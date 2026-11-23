"""
Phishing Email Detection Model
Classifies emails as "Phishing" or "Safe" using Scikit-learn.

Features used:
  - TF-IDF of the email text
  - URL features (count, IP-address URLs, link shorteners, '@' and hyphens in URLs)
  - Keyword count and exclamation marks
"""

import re
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# ---------- 1. Dataset (1 = Phishing, 0 = Safe) ----------
phish = [
    "Congratulations! You've won a lottery. Click here to claim: http://bit.ly/claim-now",
    "Urgent: Verify your account immediately at http://secure-login.verify-bank.com",
    "Important: update your payment information http://192.168.4.21/pay",
    "Your account is suspended! Confirm your password http://paypa1-support.com/login",
    "You have won a free iPhone. Click http://free-prizes.xyz now!!!",
    "Security alert: unusual login detected. Verify now http://tinyurl.com/secure-acc",
    "Dear customer, your bank account will be closed. Update details http://bank-update.info",
    "Claim your refund now! Enter card details at http://refund-gov.top/claim",
    "URGENT!!! Your password expires today. Login http://mail-support@secure.ru",
    "Final notice: unpaid invoice. Download attachment http://invoice-docs.biz/file",
    "Your Netflix payment failed. Update billing http://netflix-billing.co/update",
    "Act now: limited offer, send your OTP to confirm your prize",
]
safe = [
    "Your invoice from our recent purchase is attached.",
    "Hello! Just checking in to see how you're doing.",
    "Meet your new colleague at the office.",
    "Team meeting is scheduled for 3 PM tomorrow in room 204.",
    "Please find the project report attached. See https://company.com/reports",
    "Lunch on Friday? Let me know what works for you.",
    "The syllabus for next semester is available at https://university.edu/syllabus",
    "Thanks for your help with the presentation yesterday.",
    "Reminder: submit your timesheet by end of day.",
    "Happy birthday! Hope you have a wonderful day.",
    "Here are the notes from today's class. Let me know if you have questions.",
    "Your order has shipped and will arrive on Monday.",
]
df = pd.DataFrame({"text": phish + safe,
                   "label": [1] * len(phish) + [0] * len(safe)})

# ---------- 2. Feature extraction (URLs, keywords, etc.) ----------
KEYWORDS = ["urgent", "verify", "password", "account", "click", "winner", "won",
            "free", "prize", "suspended", "confirm", "update", "otp", "refund",
            "login"]
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co"]


def extract_features(text):
    t = text.lower()
    urls = re.findall(r"https?://\S+", t)
    return {
        "num_urls": len(urls),
        "has_ip_url": int(any(re.search(r"https?://\d+\.\d+\.\d+\.\d+", u) for u in urls)),
        "has_shortener": int(any(s in u for u in urls for s in SHORTENERS)),
        "has_at_in_url": int(any("@" in u for u in urls)),
        "url_has_hyphen": int(any("-" in u for u in urls)),
        "keyword_count": sum(k in t for k in KEYWORDS),
        "exclamations": text.count("!"),
    }


feat_df = pd.DataFrame([extract_features(t) for t in df["text"]])

# ---------- 3. Train/test split ----------
X_train_txt, X_test_txt, X_train_f, X_test_f, y_train, y_test = train_test_split(
    df["text"], feat_df, df["label"],
    test_size=0.25, random_state=42, stratify=df["label"])

# ---------- 4. Combine text (TF-IDF) + URL/keyword features ----------
tfidf = TfidfVectorizer(stop_words="english")
X_train = hstack([tfidf.fit_transform(X_train_txt), csr_matrix(X_train_f.values)])
X_test = hstack([tfidf.transform(X_test_txt), csr_matrix(X_test_f.values)])

# ---------- 5. Train & predict ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------- 6. Results ----------
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=["Safe", "Phishing"], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=["Safe", "Phishing"]).plot(cmap="Blues")
plt.title("Phishing Email Detection")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# ---------- 7. Try a new email ----------
def classify(email):
    feats = pd.DataFrame([extract_features(email)]).values
    x = hstack([tfidf.transform([email]), csr_matrix(feats)])
    return "Phishing" if model.predict(x)[0] == 1 else "Safe"


print("\nTest predictions:")
print("Urgent! Verify your password at http://bit.ly/abc123 ->",
      classify("Urgent! Verify your password at http://bit.ly/abc123"))
print("Can we move our meeting to 4 PM? ->",
      classify("Can we move our meeting to 4 PM?"))
