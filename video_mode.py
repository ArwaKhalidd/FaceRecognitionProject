import cv2
from recognizer import recognize_faces

video_path = "Lionel Messi has been called the GOAT, but here are the athletes he considers to be among the best.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("Cannot open video")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    result = recognize_faces(frame)

    cv2.imshow("Video Recognition", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()