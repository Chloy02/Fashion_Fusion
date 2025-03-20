import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
import os
import json
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import random
from datetime import datetime
from typing import Dict, List

# Set up the Streamlit page config
st.set_page_config(page_title="FashionFusion", layout="wide")

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ===================== Load External Configuration Files =====================
CONFIG_DIR = Path("config")
with open(CONFIG_DIR / "fashion_config.json", "r") as f:
    fashion_config = json.load(f)

with open(CONFIG_DIR / "deepfashion_config.json", "r") as f:
    deepfashion_config = json.load(f)

# Extract configurations
FASHION_CATEGORIES = fashion_config["FASHION_CATEGORIES"]
STYLE_RECOMMENDATIONS = fashion_config["STYLE_RECOMMENDATIONS"]
DEEPFASHION_CONFIG = deepfashion_config

# ===================== DeepFashion Dataset Class =====================
class DeepFashionDataset:
    def __init__(self, config=DEEPFASHION_CONFIG):
        self.config = config
        self.base_dir = Path(config['base_dir'])
        self.img_dir = self.base_dir / config['img_dir']
        self.annotations = self._load_annotations()
        self.category_mapping = self._create_category_mapping()

    def _load_annotations(self):
        with open(self.base_dir / self.config['annotation_file']) as f:
            return json.load(f)

    def _create_category_mapping(self):
        mapping = {}
        for main_cat, sub_cats in self.config['categories'].items():
            for sub_cat in sub_cats:
                mapping[sub_cat] = main_cat
        return mapping

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = np.array(img)
        return preprocess_input(img_array)

    def get_batch(self, batch_size=32):
        images = []
        labels = []
        category_indices = {cat: idx for idx, cat in enumerate(self.config['categories'].keys())}
        for _ in range(batch_size):
            img_info = random.choice(self.annotations['annotations'])
            img_path = self.img_dir / img_info['file_name']
            category = self.category_mapping[img_info['category_name']]
            try:
                img_processed = self.preprocess_image(img_path)
                images.append(img_processed)
                label = np.zeros(len(category_indices))
                label[category_indices[category]] = 1
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        return np.array(images), np.array(labels)

# ===================== Recommendation Functions =====================
def get_style_recommendations(category: str, 
                              confidence: float, 
                              user_preferences: Dict = None,
                              season: str = None) -> List[Dict[str, str]]:
    recommendations = []
    base_recommendations = STYLE_RECOMMENDATIONS.get(category, {})
    gender_style = user_preferences.get('gender_style', 'all')
    style_type = user_preferences.get('style', 'casual')
    if style_type in base_recommendations:
        recommendations.extend(base_recommendations[style_type].get('all', []))
        if gender_style in ['masculine', 'feminine']:
            recommendations.extend(base_recommendations[style_type].get(gender_style, []))
    return recommendations

def fetch_api_recommendations(category: str, confidence: float, endpoint: str, api_key: str) -> List[Dict[str, str]]:
    if endpoint and api_key:
        return [
            {"tip": "External API suggestion: Try a vintage look!", "occasion": "special"},
            {"tip": "External API suggestion: Consider bold prints for a party vibe!", "occasion": "party"}
        ]
    return []

def display_recommendations(category: str, confidence: float, col, api_endpoint: str, api_key: str, api_enabled: bool):
    try:
        user_prefs = {
            'style': st.session_state.get('style', 'casual'),
            'gender_style': st.session_state.get('gender_style', 'all'),
            'favorite_colors': ['blue', 'black'],
            'occasions': ['work', 'casual']
        }
        recommendations = get_style_recommendations(
            category=category,
            confidence=confidence,
            user_preferences=user_prefs,
            season=get_current_season()
        )
        if api_enabled:
            api_recs = fetch_api_recommendations(category, confidence, api_endpoint, api_key)
            recommendations.extend(api_recs)
        if not recommendations:
            col.info("No specific recommendations available for this category.")
            return
        col.subheader("Personalized Style Suggestions:")
        col.info(f"Based on your **{category.title()}**, here are some curated ideas:")
        for rec in recommendations:
            if isinstance(rec, dict):
                col.write(f"• **{rec['occasion'].title()}:** {rec['tip']}")
            else:
                col.write(f"• {rec}")
    except Exception as e:
        col.error(f"Error displaying recommendations: {str(e)}")

# ===================== Model Loading and Preprocessing for Uploaded Images =====================
@st.cache_resource
def load_model():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers[:100]:
        layer.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x2 = tf.keras.layers.Conv2D(512, (1, 1))(x)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation('relu')(x2)
    attention_weights = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x2)
    x2 = x2 * attention_weights
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(DEEPFASHION_CONFIG['categories']), activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def preprocess_uploaded_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)

def get_current_season():
    month = datetime.now().month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def process_predictions(predictions, confidence_threshold=0.1):
    category_names = list(DEEPFASHION_CONFIG['categories'].keys())
    results = []
    for idx, confidence in enumerate(predictions[0]):
        if confidence > confidence_threshold:
            category = category_names[idx]
            subcategories = DEEPFASHION_CONFIG['categories'][category]
            results.append({
                'category': category,
                'confidence': confidence,
                'subcategories': subcategories
            })
    return sorted(results, key=lambda x: x['confidence'], reverse=True)

# ===================== Session State Initialization =====================
if 'gender_style' not in st.session_state:
    st.session_state.gender_style = 'all'
if 'style' not in st.session_state:
    st.session_state.style = 'casual'

# ===================== Sidebar: User Preferences and Advanced Settings =====================
with st.sidebar:
    st.title("Style Preferences")
    st.session_state.gender_style = st.selectbox(
        "Choose your style preference",
        options=['all', 'masculine', 'feminine'],
        index=0,
        help="Select your style preference for personalized recommendations"
    )
    st.write("---")
    st.session_state.style = st.radio(
        "Style Type",
        options=['casual', 'formal'],
        index=0,
        help="Choose whether you prefer a casual or formal look"
    )
    st.write("---")
    st.header("Advanced Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Set the minimum confidence required to display a prediction."
    )
    api_enabled = st.checkbox("Enable External API Recommendations", value=False)
    api_endpoint = ""
    api_key = ""
    if api_enabled:
        api_endpoint = st.text_input("API Endpoint", value="https://example.com/api")
        api_key = st.text_input("API Key", type="password")
    st.write("---")
    st.write("### Additional Settings")
    st.write("Customize more options in future updates!")

# ===================== Main Content: Image Upload, Prediction, and Display =====================
st.title("👗 FashionFusion: Advanced Fashion Classifier")
st.markdown("Upload an image of your fashion item, and we'll classify it and give you personalized style suggestions!")

with st.spinner("Loading model... Please wait."):
    model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner('Analyzing your fashion item...'):
            progress_bar = st.progress(0)
            processed_image = preprocess_uploaded_image(image)
            progress_bar.progress(33)
            predictions = model.predict(processed_image)
            progress_bar.progress(66)
            significant = [conf for conf in predictions[0] if conf > confidence_threshold]
            if not significant:
                st.warning("Hmm... our model isn't too sure about this one. Try a clearer image!")
            progress_bar.progress(100)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Results:")
            for category, confidence in zip(FASHION_CATEGORIES.keys(), predictions[0]):
                if confidence > confidence_threshold:
                    st.success(f"🏷️ **{category.title()}**: {confidence*100:.1f}%")
                    if confidence > 0.5:
                        st.write("   _Possible subcategories:_")
                        gender_style = st.session_state.gender_style
                        subcats = FASHION_CATEGORIES[category].get('all', [])
                        if gender_style != 'all':
                            subcats.extend(FASHION_CATEGORIES[category].get(gender_style, []))
                        for subcat in subcats:
                            st.write(f"   • {subcat.title()}")
        with col2:
            top_idx = np.argmax(predictions[0])
            top_category = list(FASHION_CATEGORIES.keys())[top_idx]
            top_confidence = predictions[0][top_idx]
            display_recommendations(top_category, top_confidence, col2, api_endpoint, api_key, api_enabled)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Awaiting your fashion upload... let's see what style gem you're rocking today!")
