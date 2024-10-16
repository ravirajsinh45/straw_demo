import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import cv2

# Load the YOLO model (ensure the path to your custom model is correct)
model_path = "model/straw_16102024_small_gray.pt"  # Update with the path to your YOLO custom model
model = YOLO(model_path)

# Streamlit app
st.title("Straw Head Counting")

# Add sliders for adjusting confidence and threshold
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.4, 0.05)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Infer button
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Infer Image"):
        # Convert the image to a format suitable for the model (grayscale)
        image_np = np.array(image)
        grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

        # Convert grayscale back to 3-channel image by stacking the grayscale image
        grayscale_3ch = cv2.merge([grayscale_image] * 3)  # Stack the grayscale image into 3 channels

        # Run YOLO inference on the 3-channel grayscale image
        results = model.predict(grayscale_3ch, conf=confidence_threshold, iou=detection_threshold)

        # Get the bounding boxes
        bboxes = results[0].boxes.xyxy  # (x1, y1, x2, y2) format

        # Draw bounding boxes on the original color image
        image_np_copy = image_np.copy()
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
            cv2.rectangle(image_np_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White box with 2px thickness

        # Convert annotated image back to PIL format for Streamlit
        annotated_image = Image.fromarray(image_np_copy)

        # Display the annotated image
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        # Display the number of objects detected
        st.write(f"Number of objects detected: {len(bboxes)}")
