import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Fashion categories with more specific labels
FASHION_CATEGORIES = {
    'tops': ['t-shirt', 'shirt', 'blouse', 'tank top', 'sweater'],
    'bottoms': ['pants', 'jeans', 'shorts', 'skirt'],
    'dresses': ['dress', 'midi dress', 'maxi dress'],
    'outerwear': ['jacket', 'coat', 'blazer'],
    'footwear': ['shoes', 'boots', 'sneakers', 'sandals'],
    'accessories': ['bag', 'hat', 'scarf']
}

@st.cache_resource
def load_model():
    # Load a pre-trained ResNet50 model
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Add fashion-specific layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(FASHION_CATEGORIES), activation='softmax')
    ])
    
    return model

def preprocess_image(image):
    # Resize and preprocess image
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Streamlit UI
st.title("👗 FashionFusion: Advanced Fashion Classifier")
st.write("Upload any fashion item image for detailed classification!")

# Load model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        with st.spinner('Analyzing your fashion item...'):
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess and classify
            processed_image = preprocess_image(image)
            
            # Make prediction
            predictions = model.predict(processed_image)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Results:")
                for idx, (category, confidence) in enumerate(zip(FASHION_CATEGORIES.keys(), predictions[0])):
                    if confidence > 0.1:  # Show only if confidence > 10%
                        st.success(f"🏷️ {category.title()}: {confidence*100:.1f}%")
                        # Show subcategories if confidence is high
                        if confidence > 0.5:
                            for subcat in FASHION_CATEGORIES[category]:
                                st.write(f"   • {subcat.title()}")
            
            with col2:
                st.subheader("Style Suggestions:")
                # Add style suggestions based on the detected category
                top_category = list(FASHION_CATEGORIES.keys())[np.argmax(predictions[0])]
                st.info(f"Based on your {top_category}, we suggest:")
                if top_category == 'tops':
                    st.write("• Pair with high-waisted jeans")
                    st.write("• Layer with a blazer")
                elif top_category == 'dresses':
                    st.write("• Add a belt to define waist")
                    st.write("• Pair with statement jewelry")
                # Add more style suggestions for other categories
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.sidebar.markdown("""
### About FashionFusion
This advanced fashion classifier uses deep learning to:
- Identify clothing categories
- Suggest style combinations
- Provide fashion recommendations
""")
