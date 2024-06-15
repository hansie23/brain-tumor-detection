import streamlit as st
from unet_model import unet_model
from PIL import Image
import numpy as np

input_shape = (112, 112, 3)
weights_path = r"unet_weights.h5"

# Load the model
@st.cache_resource
def model(input_shape, weights_path):
    model = unet_model(input_shape)
    model.load_weights(weights_path)
    return model

model = model(input_shape, weights_path)

# Preprocess the image
def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((112, 112))
    new_image_array = np.array(image) / 255  # Normalize pixel values
    expanded_image_array = np.expand_dims(new_image_array, axis=0)  # Expand dimensions
    return expanded_image_array

# Predict function
def predict(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image, verbose=0)
    return prediction


# Streamlit UI
def main():
    st.title("Brain Tumor Segmentation")
    
    # Define page options
    page_options = ['Home', 'Model Interpretability']

    # Display navigation
    st.sidebar.title('Navigation')
    page_selection = st.sidebar.radio('Go to', page_options)
    
    # Display selected page
    if page_selection == 'Home':
        uploaded_image = st.file_uploader("Upload an MRI image", type=["tif"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded MRI Image', use_column_width=True)
            if st.button('Predict'):
                prediction = predict(uploaded_image, model)
                
                # Post-process prediction to display
                prediction_image = np.squeeze(prediction)  # Remove batch dimension
                prediction_image = (prediction_image * 255).astype(np.uint8)  # Scale back to [0, 255]
                prediction_image = Image.fromarray(prediction_image)
                
                with col2:
                    st.image(prediction_image, caption='Predicted Brain Tumor', use_column_width=True)

    elif page_selection == 'Model Interpretability':
        st.image('interpretability.png', use_column_width=True)

if __name__ == '__main__':
    main()
