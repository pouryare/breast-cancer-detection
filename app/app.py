import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better spacing and layout
st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_cached():
    """Load the trained model with caching."""
    model_path = os.path.join(os.getcwd(), 'breast_cancer_model.keras')
    try:
        return load_model(model_path)
    except:
        st.error("Model file not found. Please ensure 'breast_cancer_model.keras' is in the current directory.")
        return None

@st.cache_data
def preprocess_image(_image, target_size=(224, 224)):
    """Preprocess the uploaded image for model prediction."""
    try:
        # Convert to RGB if necessary
        if _image.mode != 'RGB':
            _image = _image.convert('RGB')
        
        # Resize image
        image = _image.resize(target_size, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, image_array):
    """Make prediction using the model."""
    try:
        with st.spinner('Analyzing image...'):
            # Add progress bar for visual feedback
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            prediction = model.predict(image_array)
            return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def plot_prediction_confidence(prediction):
    """Create a confidence plot for the prediction."""
    fig, ax = plt.subplots(figsize=(8, 3))
    classes = ['Benign', 'Malignant']
    colors = ['#2ecc71' if p < 0.5 else '#e74c3c' for p in prediction]
    
    sns.barplot(x=classes, y=prediction, palette=colors, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Prediction Confidence')
    ax.set_ylabel('Probability')
    
    return fig

def main():
    st.title("Breast Cancer Detection System")
    st.divider()
    
    # Load model
    model = load_model_cached()
    if model is None:
        return
    
    # Description
    st.write("""
    This system analyzes histopathological images to detect breast cancer.
    Upload an image to get started.
    """)
    st.divider()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a histopathological image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.divider()
        
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return
        
        # Make prediction
        prediction = predict_image(model, processed_image)
        if prediction is None:
            return
        
        # Display results
        st.subheader("Analysis Results")
        
        # Calculate confidence percentages
        benign_conf = (1 - prediction[1]) * 100
        malignant_conf = prediction[1] * 100
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Benign Probability",
                value=f"{benign_conf:.1f}%",
                delta="Lower risk" if benign_conf > 50 else "Higher risk"
            )
        with col2:
            st.metric(
                label="Malignant Probability",
                value=f"{malignant_conf:.1f}%",
                delta="Higher risk" if malignant_conf > 50 else "Lower risk",
                delta_color="inverse"
            )
        
        # Plot confidence
        st.divider()
        st.subheader("Confidence Visualization")
        confidence_plot = plot_prediction_confidence(prediction)
        st.pyplot(confidence_plot)
        
        # Final assessment
        st.divider()
        if malignant_conf > 50:
            st.warning("The image shows characteristics of malignant tissue. Please consult with a healthcare professional for proper diagnosis.")
        else:
            st.success("The image shows characteristics of benign tissue. However, always consult with a healthcare professional for proper diagnosis.")

if __name__ == "__main__":
    main()