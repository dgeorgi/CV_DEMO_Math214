import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2

# Initialize the main window
root = tk.Tk()
root.title("Facial Recognition Demo")
root.geometry("800x600")

# Initialize OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from the webcam (you might need to install OpenCV's VideoCapture)
cap = cv2.VideoCapture(0)

# Create a function to update the displayed image
def update_image():
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw rectangles around detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the frame to RGB for displaying with matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame using matplotlib
        ax.clear()
        ax.imshow(frame_rgb)
        canvas.draw()

# Create a canvas for displaying the webcam feed
fig = Figure(figsize=(6, 6), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Create a button to start facial recognition
start_button = tk.Button(root, text="Start Facial Recognition", command=update_image)
start_button.pack()

# Create an axes object for displaying the webcam feed
ax = fig.add_subplot(111)

# Start the main loop
root.mainloop()

# Release the webcam and close the OpenCV window when the application is closed
cap.release()
cv2.destroyAllWindows()
