import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
import os, sys
from roboflow import Roboflow
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



try:
    try:
        api_key = os.environ['ROBOFLOW_API_KEY'] 
    except:
        api_key=None
        
    if not api_key:
        api_key = st.secrets['ROBOFLOW_API_KEY']
        if not api_key:
            st.warning("enter valid key!!")
            sys.exit()

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("straw-detenction")
    model = project.version(1).model

except:
    st.warning("enter valid key!!")
    sys.exit()





# Function to make predictions using the Roboflow model
def predict_image(image_content, confidence=40, overlap=30):
    # Infer on a local image
    # image = Image.open(BytesIO(image_content))
    
    # # Convert the image to bytes
    # image_bytes = BytesIO()
    # image.save(image_bytes, format="JPEG")
    
    # Make predictions using the Roboflow model
    predictions = model.predict(image_content, confidence=confidence, overlap=overlap).json()
    return predictions

# Function to make predictions using the Roboflow model with an image URL
def predict_image_url(image_url, confidence=40, overlap=30):
    # Infer on an image hosted elsewhere
    predictions = model.predict(image_url, hosted=True, confidence=confidence, overlap=overlap).json()
    return predictions

# Function to draw rectangles on the image based on predictions
def draw_rectangles(image, predictions):
    draw = ImageDraw.Draw(image)
    
    for prediction in predictions.get("predictions", []):
        center_x, center_y, width, height = (
            prediction["x"],
            prediction["y"],
            prediction["width"],
            prediction["height"]
        )
        
        x, y = center_x - width / 2, center_y - height / 2
        draw.rectangle([x, y, x + width, y + height], outline="yellow", width=1)
    
    return image

# Streamlit app
def main():
    try:
        st.title("Straw Detection App")

        # Choose detection method
        detection_method = st.radio("Choose detection method:", ["Image Upload", "Image URL"])

        if detection_method == "Image Upload":
            # File uploader for image upload
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                width, height = image.size 


                if (width>1000) and (height>1000):
                    image = image.resize((1000,1000))                               

                temp_file_path = "temp" + str(datetime.datetime.now().timestamp()).replace(".","-") + ".jpeg"
                
                image.save(temp_file_path, format="JPEG")
                # st.image(image, caption="Uploaded Image.", use_column_width=True)

            
                # Confidence and overlap sliders
                confidence = 60 # st.slider("Confidence", min_value=0, max_value=100, value=40)
                overlap = 60 #st.slider("Overlap", min_value=0, max_value=100, value=30)

            
                # Make predictions using the Roboflow model
                predictions = predict_image(temp_file_path, confidence, overlap)

                # Draw rectangles on the image based on predictions
                annotated_image = draw_rectangles(image.copy(), predictions)

                st.subheader("Object Detection Results:")
                

                # Display object count
                if "predictions" in predictions:
                    object_count = len(predictions["predictions"])
                    st.success(f"Number of objects detected: {object_count}")
                    st.image(annotated_image, caption="Result Image.", use_column_width=True)

                else:
                    st.warning("No predictions found.")
                os.remove(temp_file_path)

        elif detection_method == "Image URL":
            # Text input for image URL
            image_url = st.text_input("Enter Image URL:")

            if st.button("Predict"):
                # Make predictions using the Roboflow model with an image URL
                predictions = predict_image_url(image_url)

                # Display the results
                # st.subheader("Object Detection Results:")
                # st.json(predictions)

                # Display object count
                if "predictions" in predictions:
                    object_count = len(predictions["predictions"])
                    st.success(f"Number of objects detected: {object_count}")
                else:
                    st.warning("No predictions found.")
    except:
        try:
            os.remove(temp_file_path)
        except:
            pass


if __name__ == "__main__":
    try:
        main()
    except:
        st.warning("Internal error!!")