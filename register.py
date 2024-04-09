import tkinter as tk
import cv2
import os

# Create a directory if it does not exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Capture images from webcam and save
def capture_images(name):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a directory with the name under "dataset" folder
    dataset_dir = "dataset"
    create_directory(dataset_dir)

    # Directory for the current person
    person_dir = os.path.join(dataset_dir, name)
    create_directory(person_dir)

    # Counter for images captured
    count = 0

    # Capture images
    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale
       # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop the face region
            face_roi = frame[y:y+h, x:x+w]

            # Resize the face region to 224x224 pixels
            resized_face = cv2.resize(face_roi, (224, 224))

            # Save the cropped and resized face image
            image_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(image_path, resized_face)

            count += 1

            # Stop capturing after 200 images
            if count == 25:
                break

            # Display the frame with the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capture Images', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count == 200:
            break

    cap.release()
    cv2.destroyAllWindows()

# Create UI
def start_capture():
    name = name_entry.get()
    if name:
        capture_images(name)
        status_label.config(text="Images captured successfully.")
    else:
        status_label.config(text="Please enter a name.")

root = tk.Tk()
root.title("Face Image Capture")

name_label = tk.Label(root, text="Enter Name:")
name_label.pack()

name_entry = tk.Entry(root)
name_entry.pack()

capture_button = tk.Button(root, text="Capture Images", command=start_capture)
capture_button.pack()

status_label = tk.Label(root, text="")
status_label.pack()

root.mainloop()
