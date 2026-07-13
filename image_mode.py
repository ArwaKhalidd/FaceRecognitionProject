import cv2
from recognizer import recognize_faces

image_path = "test.jpg"

frame = cv2.imread(image_path)

if frame is None:

    print("Cannot load image")
    exit()

result = recognize_faces(frame)

cv2.imshow("Image Recognition", result)

cv2.waitKey(0)

cv2.destroyAllWindows()