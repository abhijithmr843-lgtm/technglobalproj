# Test the trained classifier on new email/message samples
# Run spam_classifier.py first to generate model.pkl and vectorizer.pkl

import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# same cleaning function as in spam_classifier.py
def clean_text(msg):
    msg = msg.lower()
    msg = msg.translate(str.maketrans('', '', string.punctuation))
    words = []
    for w in msg.split():
        if w not in stop_words:
            words.append(stemmer.stem(w))
    return ' '.join(words)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)


def predict(msg):
    cleaned = clean_text(msg)
    vec = tfidf.transform([cleaned])
    result = model.predict(vec)[0]
    return "SPAM" if result == 1 else "HAM (not spam)"


# some test samples I made up to check the model
test_samples = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!",
    "Hey are we still meeting for lunch tomorrow?",
    "URGENT! Your account has been suspended. Verify your details immediately to avoid closure",
    "Can you send me the notes from yesterdays class",
    "FREE entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 87121",
    "Mom said dinner is at 8, dont be late",
]

print("Testing classifier on new samples:")
print("-" * 60)
for msg in test_samples:
    print(f"[{predict(msg)}]  {msg[:50]}...")

# also let the user type their own message
print()
print("Type a message to classify (or just press enter to quit):")
while True:
    msg = input("> ")
    if msg.strip() == "":
        break
    print(predict(msg))
