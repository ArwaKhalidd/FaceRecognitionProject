import cv2
import dlib
import numpy as np
import os
import pickle

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

print("Loading models...")

hog_face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor(
    os.path.join(BASE_PATH, "models", "shape_predictor_68_face_landmarks.dat")
)

face_rec_model = dlib.face_recognition_model_v1(
    os.path.join(BASE_PATH, "models", "dlib_face_recognition_resnet_model_v1.dat")
)

print("Models loaded successfully")

known_encodings = []
known_names = []

images_path = os.path.join(BASE_PATH, "images")

for file in os.listdir(images_path):

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(images_path, file)

    img = cv2.imread(path)

    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = hog_face_detector(rgb, 1)

    if len(faces) == 0:
        print(f"No face found in {file}")
        continue

    shape = shape_predictor(rgb, faces[0])

    encoding = face_rec_model.compute_face_descriptor(rgb, shape)

    known_encodings.append(np.array(encoding))

    name = os.path.splitext(file)[0]

    known_names.append(name)

    print(f"Saved: {name}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("face_vectors.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nVectors file created successfully")