# Machine Learning with Python - Projects

This repository contains my two projects for **Machine Learning with Python**.

| Project | Type | Folder |
|---------|------|--------|
| Spam Email Classifier | Minor Project | [`spam-email-classifier/`](spam-email-classifier/) |
| Face Recognition Attendance System | Major Project | [`face-attendance-system/`](face-attendance-system/) |

Each folder has its own detailed `README.md`, source code and `requirements.txt`. This file is just an overview of both.

---

## 1. Spam Email Classifier (Minor Project)

A classifier that predicts whether a message/email is **spam** or **not spam (ham)**.

**How it works:** the text is cleaned (lowercase, remove punctuation and stopwords, stemming), converted into numeric features using **TF-IDF**, and then a machine learning model predicts the class. I trained and compared three models - Naive Bayes, Logistic Regression and SVM - and tuned the best one.

**Dataset:** SMS Spam Collection (5572 labelled messages), included in `spam-email-classifier/data/`.

**Result:** the tuned Linear SVM gave the best performance - about **99% accuracy** and an **F1-score of 0.95** for spam.

**Run it:**
```
cd spam-email-classifier
pip install -r requirements.txt
python spam_classifier.py     # trains the models and saves the best one
python predict.py             # test on new messages
```
You can also open `spam_classifier.ipynb` in Jupyter to see the data exploration with charts.

**Tools:** Python, Pandas, NumPy, Scikit-learn, NLTK, Jupyter Notebook

---

## 2. Face Recognition Attendance System (Major Project)

An attendance system that recognizes a student's face from the webcam and automatically marks them present with the date and time.

**How it works:** faces are detected with OpenCV's **Haar Cascade** and recognized with the **LBPH** (Local Binary Patterns Histograms) recognizer. You register each student once by capturing their face images, train the model, and then the system recognizes them live and writes attendance to a CSV file (only once per day). A small Tkinter dashboard shows the records.

**Dataset:** the system builds its own dataset from the webcam, so no external dataset is needed.

**Run it (needs a webcam):**
```
cd face-attendance-system
pip install -r requirements.txt
python capture_faces.py       # register each student (enter ID + name)
python train_model.py         # train the recognizer
python attendance.py          # mark attendance live, press Q to quit
python dashboard.py           # view the attendance records
```

**Tools:** Python, OpenCV (opencv-contrib-python), NumPy, Tkinter, Haar Cascade, LBPH recognizer

---

## Requirements

- Python 3.x
- The libraries listed in each project's `requirements.txt`
- A webcam (only for the Face Recognition project)

## Notes

- The spam classifier runs fully on any machine.
- The face attendance system needs a webcam for capturing and marking attendance, and a desktop to show the dashboard window.
