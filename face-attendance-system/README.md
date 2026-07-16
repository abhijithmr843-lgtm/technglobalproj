# Face Recognition Attendance System

Major project for **Machine Learning with Python**.

An attendance system that uses face recognition to automatically mark students present. You register a student's face once, and after that the system recognizes them from the webcam and records their attendance with the date and time.

## How it works

1. **Capture** - take ~50 photos of each student's face using the webcam
2. **Train** - train an LBPH face recognizer on those photos
3. **Recognize** - open the webcam, detect faces and match them against the trained model
4. **Attendance** - when a known face is recognized, mark them present in a CSV file (only once per day)
5. **Dashboard** - view all attendance records in a small GUI window

For face **detection** I used OpenCV's Haar Cascade classifier, and for face **recognition** I used the LBPH (Local Binary Patterns Histograms) recognizer from OpenCV. LBPH works well even with a small number of training images, which is good for a student project.

## Project files

| File | What it does |
|------|--------------|
| `capture_faces.py` | Captures face images from the webcam and saves them into `dataset/` |
| `train_model.py` | Trains the LBPH recognizer, creates `trainer.yml` and `labels.pkl` |
| `attendance.py` | Recognizes faces live and marks attendance in `attendance.csv` |
| `dashboard.py` | Tkinter window to view the attendance records |
| `requirements.txt` | Libraries needed |

## How to run

1. Install the requirements:
```
pip install -r requirements.txt
```

2. **Register each student** (run once per student, enter their ID and name):
```
python capture_faces.py
```
This saves the images into `dataset/<id>_<name>/`.

3. **Train the model** after all students are registered:
```
python train_model.py
```

4. **Take attendance** - this opens the webcam and marks present anyone it recognizes:
```
python attendance.py
```
Press **Q** to stop. Attendance gets saved to `attendance.csv`.

5. **View the dashboard**:
```
python dashboard.py
```

## Folder structure after running

```
face-attendance-system/
├── capture_faces.py
├── train_model.py
├── attendance.py
├── dashboard.py
├── dataset/                 <- created by capture_faces.py
│   ├── 101_John_Doe/
│   └── 205_Jane_Smith/
├── trainer.yml              <- created by train_model.py
├── labels.pkl               <- created by train_model.py
└── attendance.csv           <- created by attendance.py
```

## Tools used

Python, OpenCV (opencv-contrib-python), NumPy, Tkinter, Haar Cascade, LBPH recognizer

## Notes / things I learned

- Haar Cascade is fast for detecting faces but the LBPH recognizer needs the face cropped and in grayscale
- More training images per person = better and more stable recognition
- The `confidence` value from LBPH is actually a *distance*, so a **lower** number means a **better** match. I set a threshold of 70 - anything above that is treated as "Unknown"
- Good, even lighting during capture makes a big difference in accuracy
- I originally wanted to use the `face_recognition` (dlib) library but it was hard to install on Windows, and OpenCV's LBPH gave good results with much less setup

## Possible improvements

- Use deep learning embeddings (FaceNet / Dlib) with an SVM/KNN classifier for higher accuracy
- Add an anti-spoofing check so a photo cant be used to mark attendance
- Export attendance to Excel and show weekly/monthly reports in the dashboard
- Handle multiple faces in the frame at the same time (already partly supported)

## Dataset

The system builds its own dataset from your webcam, so no external dataset is required. For reference, public face datasets like the ones below can also be used for testing:
- https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- https://www.kaggle.com/datasets/andrewmvd/faces-in-the-wild
