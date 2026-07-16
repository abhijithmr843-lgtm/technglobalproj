# Step 1 - Capture face images of a student using the webcam
# Run this once for each student. It saves ~50 face images into
# dataset/<id>_<name>/ which are later used to train the recognizer.

import cv2
import os

# haar cascade for face detection (comes bundled with opencv)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

DATASET_DIR = 'dataset'
NUM_IMAGES = 50   # how many face images to capture per student


def main():
    # get student details
    student_id = input("Enter student ID (roll no): ").strip()
    name = input("Enter student name: ").strip().replace(" ", "_")

    # make a folder for this student
    person_dir = os.path.join(DATASET_DIR, f"{student_id}_{name}")
    os.makedirs(person_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: could not open webcam")
        return

    print("Look at the camera. Capturing faces... press Q to stop early.")
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            # crop just the face and save it
            face_img = gray[y:y + h, x:x + w]
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            # draw a rectangle + count on the live video
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{NUM_IMAGES}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capturing Faces - press Q to quit", frame)

        # stop when we have enough images or user presses q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= NUM_IMAGES:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {count} images to {person_dir}")


if __name__ == '__main__':
    main()
