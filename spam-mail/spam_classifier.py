Steps covered:
1. Define objective
2. Load dataset
3. Preprocess text
4. TF-IDF feature extraction
5. Train/test split
6-7. Train + tune Naive Bayes, Logistic Regression, SVM
8. Evaluate models
9. Test on new sample messages
10. Print methodology/summary


import os
import re
import string
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_PATH = "spam.csv"
def load_dataset(path=DATA_PATH):
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin-1")
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]
        print(f"[INFO] Loaded real dataset from '{path}' with {len(df)} rows.")
    else:
        print(f"[INFO] '{path}' not found. Using a small built-in demo dataset.")
        print("[INFO] Download the real dataset from Kaggle and place it in this folder for your final run.")
        demo_data = {
            "label": ["ham", "spam"] * 10,
            "message": [
                "Hey, are we still on for lunch tomorrow?",
                "WINNER!! You have been selected to receive a FREE iPhone. Click here now!!!",
                "Can you send me the notes from today's class?",
                "URGENT: Your account has been suspended. Verify your details immediately.",
                "Happy birthday! Hope you have a great day.",
                "Congratulations! You won a $1000 Walmart gift card. Claim now at bit.ly/xyz",
                "Let's meet at the library at 5pm.",
                "FREE entry in a weekly competition to win an iPad. Text WIN to 80085 now!",
                "Don't forget to bring your laptop for the meeting.",
                "You have been selected for a cash prize of Rs. 50000. Reply to claim.",
                "Mom asked if you're coming home for dinner tonight.",
                "Limited time offer! Buy one get one FREE. Shop now before it's gone!",
                "The project deadline has been extended to Friday.",
                "You've won a lottery of $5 million! Send your bank details to claim.",
                "Thanks for helping me move last weekend, I owe you one.",
                "Get cheap loans instantly, no credit check required. Apply now!",
                "See you at the gym at 7am.",
                "Click this link to unlock exclusive adult content, 100% free!",
                "I'll call you once I land at the airport.",
                "Your parcel is on hold. Pay a small fee to release it now.",
            ],
        }
        df = pd.DataFrame(demo_data)

    df = df.dropna(subset=["message"])
    df["label"] = df["label"].str.strip().str.lower()
    return df
def setup_nltk():
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def preprocess_text(text, stop_words):
    from nltk.tokenize import word_tokenize

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)
def main():
    print("=" * 70)
    print("STEP 1: OBJECTIVE")
    print("=" * 70)
    print("Classify SMS/email messages as SPAM or HAM (not spam) using ML.\n")

    print("=" * 70)
    print("STEP 2: LOAD DATASET")
    print("=" * 70)
    df = load_dataset()
    print(df["label"].value_counts())
    print(f"Total messages: {len(df)}\n")

    print("=" * 70)
    print("STEP 3: PREPROCESS TEXT")
    print("=" * 70)
    setup_nltk()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))
    df["clean_message"] = df["message"].apply(lambda x: preprocess_text(x, stop_words))
    print(df[["message", "clean_message"]].head(5).to_string(index=False))
    print()

    print("=" * 70)
    print("STEP 4: TF-IDF FEATURE EXTRACTION")
    print("=" * 70)
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["clean_message"])
    y = df["label"].map({"ham": 0, "spam": 1})
    print(f"Feature matrix shape: {X.shape}\n")

    print("=" * 70)
    print("STEP 5: TRAIN/TEST SPLIT")
    print("=" * 70)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")

    print("=" * 70)
    print("STEP 6-7: TRAIN & TUNE MODELS")
    print("=" * 70)
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    models_and_grids = {
        "Naive Bayes": (MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]}),
        "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
        "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    }

    best_models = {}
    for name, (model, params) in models_and_grids.items():
        grid = GridSearchCV(model, params, cv=5, scoring="f1")
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"{name}: best params = {grid.best_params_}, best CV F1 = {grid.best_score_:.4f}")
    print()

    print("=" * 70)
    print("STEP 8: EVALUATE MODELS")
    print("=" * 70)
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
    )

    results = []
    for name, model in best_models.items():
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-score": f1_score(y_test, preds),
        })

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    print(results_df.to_string(index=False))
    print()

    best_model_name = results_df.iloc[0]["Model"]
    best_model = best_models[best_model_name]
    preds = best_model.predict(X_test)

    print(f"Best performing model: {best_model_name}\n")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[INFO] Confusion matrix saved as 'confusion_matrix.png'\n")

    print("=" * 70)
    print("STEP 9: TEST ON NEW SAMPLES")
    print("=" * 70)

    def predict_message(text):
        cleaned = preprocess_text(text, stop_words)
        vec = tfidf.transform([cleaned])
        pred = best_model.predict(vec)[0]
        return "SPAM" if pred == 1 else "HAM (Not Spam)"

    new_samples = [
        "Congratulations! You have won a free cruise trip. Call now to claim your prize!",
        "Hey, can we reschedule our meeting to 3pm tomorrow?",
        "URGENT: Verify your bank account now to avoid suspension.",
        "Don't forget mom's birthday is this weekend.",
        "You've been selected for a $10,000 cash reward, click the link below!",
    ]

    for msg in new_samples:
        print(f"Message: {msg}")
        print(f"Prediction: {predict_message(msg)}\n")

    print("=" * 70)
    print("STEP 10: METHODOLOGY & SUMMARY")
    print("=" * 70)
    print("""
Methodology:
1. Dataset: SMS Spam Collection (ham/spam labeled messages).
2. Preprocessing: lowercasing, URL/number/punctuation removal, stopword removal, tokenization.
3. Feature extraction: TF-IDF with unigrams+bigrams, top 3000 features.
4. Modeling: Naive Bayes, Logistic Regression, SVM - each tuned with 5-fold GridSearchCV (F1 scoring).
5. Evaluation: Accuracy, Precision, Recall, F1-score, confusion matrix on 20% held-out test set.

Insights:
- Spam messages have distinctive vocabulary ('free', 'win', 'urgent', 'claim') well captured by TF-IDF.
- Class imbalance means F1-score/recall matter more than raw accuracy.
- Bigrams help catch spam phrasing patterns beyond single words.
""")
    print("Done. See 'confusion_matrix.png' in this folder for the chart.")


if __name__ == "__main__":
    main()
