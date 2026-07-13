import cv2
from recognizer import recognize_faces

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("No camera found")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    result = recognize_faces(frame)

    cv2.imshow("Webcam Recognition", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()