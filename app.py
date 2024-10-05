import streamlit as st
from PIL import Image
import supervision as sv
from inference import get_model
import numpy as np

# Load the model
project_name = "semen-straw-counting"
version = 2
model = get_model(model_id=f"{project_name}/{version}")

# Streamlit app
st.title("Straw Head Counting")

# Add sliders for adjusting confidence and threshold
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.3, 0.05)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Infer button
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Infer Image"):
        # Run inference with the adjusted thresholds
        results = model.infer(image, confidence=confidence_threshold, threshold=detection_threshold)[0]
        detections = sv.Detections.from_inference(results)

        # Annotate the image with bounding boxes and labels
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        # annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Convert to displayable format
        annotated_image = np.array(annotated_image)

        # Display the annotated image
        st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        # Display number of objects detected
        st.write(f"Number of objects detected: {len(detections)}")
