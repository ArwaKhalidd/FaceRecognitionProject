import cv2
import dlib
import numpy as np
import os
import pickle

# --------------------------------
# PROJECT PATH
# --------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# --------------------------------
# LOAD MODELS
# --------------------------------
print("Loading models...")

hog_face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor(
    os.path.join(BASE_PATH, "models", "shape_predictor_68_face_landmarks.dat")
)

face_rec_model = dlib.face_recognition_model_v1(
    os.path.join(BASE_PATH, "models", "dlib_face_recognition_resnet_model_v1.dat")
)

print("Models loaded successfully")

# --------------------------------
# LOAD VECTORS
# --------------------------------
with open(os.path.join(BASE_PATH, "face_vectors.pkl"), "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f"{len(known_names)} faces loaded")

# --------------------------------
# FACE ENCODING
# --------------------------------
def get_face_encoding(image, face):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    shape = shape_predictor(rgb, face)

    encoding = face_rec_model.compute_face_descriptor(rgb, shape)

    return np.array(encoding)

# --------------------------------
# RECOGNITION
# --------------------------------
def recognize_faces(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = hog_face_detector(rgb, 0)

    for face in faces:

        encoding = get_face_encoding(frame, face)

        name = "Unknown"

        distances = np.linalg.norm(
            np.array(known_encodings) - encoding,
            axis=1
        )

        best_match = np.argmin(distances)

        if distances[best_match] < 0.52:
            name = known_names[best_match]

        x = max(0, face.left())
        y = max(35, face.top())

        w = face.width()
        h = face.height()

        color = (0, 255, 0)

        if name == "Unknown":
            color = (0, 0, 255)

        # rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # label background
        cv2.rectangle(
            frame,
            (x, y - 30),
            (x + w, y),
            color,
            -1
        )

        # name
        cv2.putText(
            frame,
            name,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame