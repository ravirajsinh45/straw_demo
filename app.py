import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import cv2

# Load the YOLO model (ensure the path to your custom model is correct)
roi_model_path = "model/straw_roi_only_24102024.pt"  # Update with the path to your YOLO custom model
roi_model = YOLO(roi_model_path)

straw_model_path = "model/straw_15022025_gray_736.pt"
straw_model = YOLO(straw_model_path)

# Streamlit app
st.title("Straw Head Counting")

# Add sliders for adjusting confidence and threshold
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
detection_threshold = 0.6 #st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Infer button
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Infer Image"):
        # Convert the image to numpy array
        image_np = np.array(image)

        # Run YOLO inference to detect ROI and straws
        results = roi_model.predict(image_np, conf=0.3, iou=0.5)

        print(len(results[0].boxes))

        # Filter the detections for ROI first
        rois = [box for box in results[0].boxes if int(box.cls) == 0]  # Assuming class '1' is the ROI

        if len(rois) > 0:
            # Take the first ROI detected (or handle multiple if necessary)
            roi_bbox = rois[0].xyxy[0]  # Extract the bounding box for ROI (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, roi_bbox)  # Convert to integers

            # Crop the image based on the ROI
            cropped_image = image_np[y1:y2, x1:x2]

            # Convert the cropped image to grayscale
            grayscale_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

            # Convert grayscale to 3-channel image to make it compatible with the model
            grayscale_3ch_cropped = cv2.merge([grayscale_cropped_image] * 3)

            color_cropped_resized = cv2.resize(cropped_image,(736,736))
            grayscale_3ch_cropped_resized = cv2.resize(grayscale_3ch_cropped,(736,736))

            # Run YOLO inference again on the grayscale cropped ROI to detect straws
            cropped_results = straw_model.predict(grayscale_3ch_cropped_resized, 
                                                  conf=confidence_threshold, 
                                                  iou=detection_threshold,max_det=1000)

            # Get the bounding boxes for straws from the cropped image
            straw_bboxes = [box for box in cropped_results[0].boxes if int(box.cls) == 0]  

            # Draw bounding boxes for straws on the grayscale cropped image
            for bbox in straw_bboxes:
                sx1, sy1, sx2, sy2 = map(int, bbox.xyxy[0])
                cv2.rectangle(color_cropped_resized, (sx1, sy1), (sx2, sy2), (255, 255, 255), 2)  # White box with 2px thickness

            # Convert the cropped grayscale annotated image back to PIL format for Streamlit
            annotated_cropped_image = Image.fromarray(color_cropped_resized)

            # Display the cropped and annotated image with straws
            st.image(annotated_cropped_image, caption="Cropped Grayscale Image with Straws", use_column_width=True)

            # Display the number of straws detected
            st.write(f"Number of straws detected: {len(straw_bboxes)}")
        else:
            st.write("No ROI detected.")
