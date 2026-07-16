# Spam Email Classifier

Minor project for **Machine Learning with Python**.

A simple classifier that tells whether an email/message is **spam** or **not spam (ham)**. Built with Python and scikit-learn.

## What it does

It takes the text of a message, cleans it up, converts it into numbers using TF-IDF, and then a machine learning model predicts if it is spam. I trained 3 different models and compared them to pick the best one.

## Dataset

SMS Spam Collection dataset (5572 labelled messages).
- Kaggle link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- I actually downloaded the original from the UCI repository, it is the exact same data. It is already in the `data/` folder.

The data is imbalanced - about 87% ham and 13% spam.

## Steps followed

1. Load the dataset (tab separated file)
2. Preprocess the text - lowercase, remove punctuation, remove stopwords, tokenize and stemming
3. Convert text to numeric features using **TF-IDF**
4. Split into train (80%) and test (20%) sets
5. Train 3 algorithms - **Naive Bayes**, **Logistic Regression** and **SVM**
6. Evaluate using accuracy, precision, recall and F1-score
7. Tune the best model with GridSearchCV
8. Test on new messages

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 0.975 | 1.00 | 0.81 | 0.896 |
| Logistic Regression | 0.962 | 1.00 | 0.72 | 0.836 |
| SVM (Linear) | 0.986 | 0.985 | 0.906 | 0.944 |
| **SVM tuned (C=2.0)** | **0.99** | **0.99** | **0.91** | **0.95** |

The tuned Linear SVM performed the best. Naive Bayes had perfect precision (never wrongly flagged a real message) but lower recall so it missed more spam.

## How to run

1. Install the requirements:
```
pip install -r requirements.txt
```

2. Download NLTK stopwords (only needed the first time):
```
python -c "import nltk; nltk.download('stopwords')"
```

3. Train the models (this creates `model.pkl` and `vectorizer.pkl`):
```
python spam_classifier.py
```

4. Test the classifier on new messages:
```
python predict.py
```

You can also open `spam_classifier.ipynb` in Jupyter Notebook to see the whole thing step by step with charts.

## Files

- `spam_classifier.py` - trains and compares the models, saves the best one
- `predict.py` - loads the saved model and classifies new messages
- `spam_classifier.ipynb` - notebook version with data exploration and plots
- `data/SMSSpamCollection` - the dataset
- `requirements.txt` - python libraries needed

## Tools used

Python, Pandas, NumPy, Scikit-learn, NLTK, Jupyter Notebook

## What I learned

- Text data needs a lot of cleaning before a model can use it
- TF-IDF is a simple but powerful way to represent text
- On imbalanced datasets accuracy can be misleading, so F1-score is more useful
- Different models behave differently - some are more "careful" (high precision), some catch more (high recall)

## Possible improvements

- Use n-grams (word pairs) in TF-IDF
- Try a larger email dataset
- Try deep learning (LSTM / BERT)
