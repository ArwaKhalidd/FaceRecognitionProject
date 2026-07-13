import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image

from recognizer import recognize_faces


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


root = ctk.CTk()

root.title("Face Recognition System")

root.geometry("1200x750")


title = ctk.CTkLabel(
    root,
    text="AI Face Recognition System",
    font=("Arial", 28, "bold")
)

title.pack(pady=15)


video_frame = ctk.CTkFrame(
    root,
    width=1000,
    height=550,
    corner_radius=20
)

video_frame.pack(pady=10)

video_frame.pack_propagate(False)

video_label = ctk.CTkLabel(
    video_frame,
    text=""
)

video_label.pack(expand=True)


cap = None
running = False


def stop_current_mode():

    global cap, running

    running = False

    if cap is not None:

        cap.release()

        cap = None

    video_label.configure(image=None)

    video_label.image = None


def process_and_show(frame):

    frame = recognize_faces(frame)

    h, w = frame.shape[:2]

    max_w = 1000
    max_h = 550

    scale = min(max_w / w, max_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(rgb)

    ctk_img = ctk.CTkImage(
        light_image=img,
        dark_image=img,
        size=(new_w, new_h)
    )

    video_label.configure(image=ctk_img)

    video_label.image = ctk_img


def show_frame():

    global cap, running

    if not running:
        return

    if cap is not None:

        ret, frame = cap.read()

        if ret:

            frame = cv2.resize(frame, (800, 600))

            process_and_show(frame)

        root.after(10, show_frame)


def open_image():

    stop_current_mode()

    path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if not path:
        return

    frame = cv2.imread(path)

    if frame is None:

        print("Cannot load image")

        return

    process_and_show(frame)


def open_video():

    global cap, running

    stop_current_mode()

    path = filedialog.askopenfilename(
        filetypes=[
            ("Video Files", "*.mp4 *.avi")
        ]
    )

    if not path:
        return

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        print("Cannot open video")

        return

    running = True

    show_frame()


def open_webcam():

    global cap, running

    stop_current_mode()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("No Camera Found")

        return

    running = True

    show_frame()


btn_frame = ctk.CTkFrame(
    root,
    fg_color="transparent"
)

btn_frame.pack(pady=20)

ctk.CTkButton(
    btn_frame,
    text="Open Image",
    command=open_image,
    width=180
).grid(row=0, column=0, padx=10)

ctk.CTkButton(
    btn_frame,
    text="Open Video",
    command=open_video,
    width=180
).grid(row=0, column=1, padx=10)

ctk.CTkButton(
    btn_frame,
    text="Open Webcam",
    command=open_webcam,
    width=180
).grid(row=0, column=2, padx=10)

ctk.CTkButton(
    btn_frame,
    text="Stop",
    command=stop_current_mode,
    width=180,
    fg_color="red"
).grid(row=0, column=3, padx=10)


root.mainloop()
