# Step 3 - Mark attendance using face recognition
# Opens the webcam, recognizes known faces and writes their attendance
# (name, date, time) into attendance.csv. Each student is only marked once
# per day.

import cv2
import os
import pickle
import csv
from datetime import datetime

ATTENDANCE_FILE = 'attendance.csv'
CONFIDENCE_THRESHOLD = 70   # lower = more confident. above this we treat as "unknown"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_recognizer():
    if not os.path.exists('trainer.yml'):
        print("trainer.yml not found. Run train_model.py first!")
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    with open('labels.pkl', 'rb') as f:
        label_map = pickle.load(f)
    return recognizer, label_map


def already_marked(student):
    """check if this student was already marked today"""
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    today = datetime.now().strftime("%Y-%m-%d")
    with open(ATTENDANCE_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] == student and row[1] == today:
                return True
    return False


def mark_attendance(student):
    if already_marked(student):
        return False
    now = datetime.now()
    # create file with header if it doesnt exist
    new_file = not os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['Name', 'Date', 'Time'])
        writer.writerow([student, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    print(f"Marked present: {student}")
    return True


def main():
    recognizer, label_map = load_recognizer()
    if recognizer is None:
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: could not open webcam")
        return

    print("Starting attendance. Press Q to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face_img)

            if confidence < CONFIDENCE_THRESHOLD:
                # label_map value looks like "101_John_Doe", clean it up for display
                raw_name = label_map.get(label, "Unknown")
                name = raw_name.split("_", 1)[-1].replace("_", " ")
                mark_attendance(raw_name)
                color = (0, 255, 0)
                text = f"{name} ({round(confidence)})"
            else:
                color = (0, 0, 255)
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Attendance saved to {ATTENDANCE_FILE}")


if __name__ == '__main__':
    main()
