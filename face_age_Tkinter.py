import tkinter as tk
from PIL import Image, ImageTk

def activation():
    import cv2

    # Load the Haar cascade classifiers for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Function to estimate age group based on face parameters
    def estimate_age_group(face_width, face_height, eye_count):
        if face_width < 100 and face_height < 100 and eye_count == 2:
            return "Child"
        elif face_width < 200 and face_height < 200 and eye_count == 2:
            return "Young Adult"
        elif face_width < 300 and face_height < 300 and eye_count == 2:
            return "Adult"
        else:
            return "Elderly"

    # Function to detect gender based on presence of eyes
    def detect_gender(eye_count):
        if eye_count == 2:
            return "Female"
        else:
            return "Male"

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Detect eyes
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Estimate age group
            age_group = estimate_age_group(w, h, len(eyes))

            # Detect gender
            gender = detect_gender(len(eyes))

            # Display age group and gender
            cv2.putText(frame, f"Age Group: {age_group}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Create the main window
root = tk.Tk()
root.title("Age and Gender Identifier")
root.geometry("700x700")
root.configure(bg="royalblue")

# Load and display the image
image_path = "facescannerpic3.jpg"
image = Image.open(image_path)
image = image.resize((600, 600), Image.ANTIALIAS)  # Resize the image to fit the window
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=photo)
image_label.pack()

# Create the button to start face detection
button = tk.Button(root, text="Face Detector", command=activation, bg="#C0C0C0", fg="black", font=("Arial", 14, "bold"), padx=10, pady=5, borderwidth=2, relief="raised", activebackground="#6495ED", activeforeground="white")
button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

# Run the Tkinter event loop
root.mainloop()
