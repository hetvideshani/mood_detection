import streamlit as st
import cv2
import numpy as np
import warnings
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from PIL import Image

warnings.filterwarnings("ignore")

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('model3.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('model3.h5')
    return model

model = load_model()

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Function to preprocess image for model
def extract_feature(face_img):
    face_img = cv2.resize(face_img, (48, 48))  # Ensure size is correct
    feature = np.array(face_img)

    if feature.size != 48 * 48:  # Validate shape
        st.error("Unexpected image shape. Cannot reshape.")
        return None

    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize pixel values

# Streamlit UI
st.title("🎭 Real-time Facial Emotion Detection")

# Capture Image
img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Convert image to OpenCV format (grayscale)
    image = Image.open(img_file_buffer).convert("L")
    open_cv_image = np.array(image)

    # Detect faces
    faces = face_cascade.detectMultiScale(open_cv_image, scaleFactor=1.32, minNeighbors=5)

    if len(faces) > 0:
        for (p, q, r, s) in faces:
            face_img = open_cv_image[q:q+s, p:p+r]

            img_pixels = extract_feature(face_img)
            if img_pixels is None:  # Skip if feature extraction failed
                continue

            # Predict emotion
            prediction = model.predict(img_pixels)
            predicted_emotion = labels[np.argmax(prediction)]

            # Draw bounding box & label
            cv2.rectangle(open_cv_image, (p, q), (p+r, q+s), (255, 0, 0), 2)
            cv2.putText(open_cv_image, predicted_emotion, (p, q-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert grayscale image to BGR before displaying
        open_cv_image_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)

        # Display processed image
        st.image(open_cv_image_bgr, channels="BGR", caption=f"Detected Emotion: {predicted_emotion}")

    else:
        st.warning("No face detected. Please try again!")
