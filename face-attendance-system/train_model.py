# Step 2 - Train the face recognizer on the captured images
# Reads every image inside the dataset/ folders, trains an LBPH face
# recognizer and saves it to trainer.yml. Also saves the id -> name map.

import cv2
import os
import numpy as np
import pickle

DATASET_DIR = 'dataset'


def get_images_and_labels():
    faces = []
    labels = []
    label_map = {}          # numeric label -> "id_name"
    current_label = 0

    # each sub folder is one student
    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = person_name

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            # read as grayscale
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            faces.append(gray)
            labels.append(current_label)

        print(f"  loaded {person_name} -> label {current_label}")
        current_label += 1

    return faces, labels, label_map


def main():
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print("No dataset found. Run capture_faces.py first!")
        return

    print("Loading images...")
    faces, labels, label_map = get_images_and_labels()
    print(f"Total images: {len(faces)}, Total students: {len(label_map)}")

    # LBPH = Local Binary Patterns Histograms, works well for face recognition
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Training...")
    recognizer.train(faces, np.array(labels))

    recognizer.save('trainer.yml')
    with open('labels.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    print("Training done. Saved trainer.yml and labels.pkl")


if __name__ == '__main__':
    main()
