# Spam Email Classifier - Minor Project
# Trains 3 models (Naive Bayes, Logistic Regression, SVM) on the SMS Spam Collection
# dataset and saves the best one to model.pkl
# Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# (I downloaded the original one from UCI, same data)

import pandas as pd
import string
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# download stopwords if not there already
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_text(msg):
    """Preprocessing: lowercase, remove punctuation, remove stopwords, stemming"""
    msg = msg.lower()
    # remove punctuation
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    # tokenize (simple split is enough here) and remove stopwords + stem
    words = []
    for w in msg.split():
        if w not in stop_words:
            words.append(stemmer.stem(w))
    return ' '.join(words)


def main():
    # 1. Load dataset
    # the file is tab separated with no header: label \t message
    df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
    print("Dataset loaded:", df.shape)
    print(df['label'].value_counts())
    print()

    # 2. Preprocess the text
    print("cleaning text... (takes a few seconds)")
    df['cleaned'] = df['message'].apply(clean_text)

    # convert labels to 0/1, spam = 1
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})

    # 3. TF-IDF features
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['target']

    # 4. train/test split - 80/20, stratify so spam ratio stays same in both
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])
    print()

    # 5. Try the 3 models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM (Linear)': LinearSVC()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = (model, f1)

        print("=" * 50)
        print(name)
        print("=" * 50)
        print("Accuracy :", round(acc, 4))
        print("Precision:", round(prec, 4))
        print("Recall   :", round(rec, 4))
        print("F1-score :", round(f1, 4))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print()

    # 6. Hyperparameter tuning on the best model
    # I expected Naive Bayes to win (most tutorials use it for spam) but SVM got
    # the best F1 score here, so tuning the C parameter for the SVM
    print("Tuning SVM C parameter with GridSearchCV...")
    params = {'C': [0.1, 0.5, 1.0, 2.0, 5.0]}
    grid = GridSearchCV(LinearSVC(), params, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    print("Best C:", grid.best_params_['C'])

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print()
    print("Final model report (tuned SVM):")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    # 7. Save model + vectorizer so predict.py can use them
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("Saved model.pkl and vectorizer.pkl")


if __name__ == '__main__':
    main()
