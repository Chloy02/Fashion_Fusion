import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
import os
from typing import Dict, List
from datetime import datetime

# Set up the Streamlit page config at the very top
st.set_page_config(page_title="FashionFusion", layout="wide")

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Define fashion categories with gender specifications
FASHION_CATEGORIES = {
    'tops': {
        'all': ['t-shirt', 'sweater'],
        'masculine': ['dress shirt', 'polo', 'henley'],
        'feminine': ['blouse', 'crop top', 'camisole'],
    },
    'bottoms': {
        'all': ['jeans', 'shorts', 'trousers'],
        'masculine': ['cargo pants', 'chinos'],
        'feminine': ['skirt', 'leggings', 'culottes'],
    },
    'dresses': {
        'all': ['jumpsuit'],
        'feminine': ['dress', 'midi dress', 'maxi dress', 'sundress'],
        'masculine': ['kaftan', 'kurta']
    },
    'outerwear': {
        'all': ['jacket', 'coat'],
        'masculine': ['bomber jacket', 'leather jacket'],
        'feminine': ['cardigan', 'shawl', 'bolero']
    },
    'footwear': {
        'all': ['sneakers', 'boots'],
        'masculine': ['oxfords', 'loafers'],
        'feminine': ['heels', 'flats', 'wedges']
    },
    'accessories': {
        'all': ['bag', 'scarf', 'sunglasses'],
        'masculine': ['tie', 'bow tie', 'wallet'],
        'feminine': ['clutch', 'statement necklace', 'earrings']
    }
}

# Define style recommendations for each category, split by casual/formal and gender
STYLE_RECOMMENDATIONS = {
    'tops': {
        'casual': {
            'all': [
                {'tip': 'Layer with a denim jacket', 'occasion': 'weekend'},
                {'tip': 'Add a statement watch', 'occasion': 'daily'}
            ],
            'masculine': [
                {'tip': 'Roll up sleeves for a casual look', 'occasion': 'daily'},
                {'tip': 'Pair with dark wash jeans', 'occasion': 'casual'}
            ],
            'feminine': [
                {'tip': 'Front-tuck with high-waisted bottoms', 'occasion': 'daily'},
                {'tip': 'Add delicate layered necklaces', 'occasion': 'casual'}
            ]
        },
        'formal': {
            'all': [
                {'tip': 'Add a blazer for instant polish', 'occasion': 'business'}
            ],
            'masculine': [
                {'tip': 'Pair with a silk tie', 'occasion': 'formal'},
                {'tip': 'Add cufflinks for elegance', 'occasion': 'business'}
            ],
            'feminine': [
                {'tip': 'Add pearl accessories', 'occasion': 'formal'},
                {'tip': 'Pair with a pencil skirt', 'occasion': 'business'}
            ]
        }
    },
    'bottoms': {
        'casual': {
            'all': [
                {'tip': 'Pair with a simple tee', 'occasion': 'casual'},
                {'tip': 'Try a relaxed fit for comfort', 'occasion': 'weekend'}
            ],
            'masculine': [
                {'tip': 'Match with a crisp button-down', 'occasion': 'daily'}
            ],
            'feminine': [
                {'tip': 'Combine with a cute blouse', 'occasion': 'casual'}
            ]
        },
        'formal': {
            'all': [
                {'tip': 'Tuck in for a polished look', 'occasion': 'business'}
            ],
            'masculine': [
                {'tip': 'Pair with a tailored jacket', 'occasion': 'formal'}
            ],
            'feminine': [
                {'tip': 'Opt for a sleek pencil skirt', 'occasion': 'formal'}
            ]
        }
    },
    'dresses': {
        'casual': {
            'all': [
                {'tip': 'Pair with sandals for a relaxed vibe', 'occasion': 'weekend'}
            ],
            'masculine': [
                {'tip': 'Try a modern twist with minimal accessories', 'occasion': 'daily'}
            ],
            'feminine': [
                {'tip': 'Add a belt to enhance your silhouette', 'occasion': 'casual'}
            ]
        },
        'formal': {
            'all': [
                {'tip': 'Accessorize with elegant jewelry', 'occasion': 'event'}
            ],
            'masculine': [
                {'tip': 'Choose classic cuts for a refined style', 'occasion': 'formal'}
            ],
            'feminine': [
                {'tip': 'Opt for high heels and statement earrings', 'occasion': 'formal'}
            ]
        }
    },
    'outerwear': {
        'casual': {
            'all': [
                {'tip': 'Layer with a light scarf', 'occasion': 'daily'}
            ],
            'masculine': [
                {'tip': 'Pair with a rugged look', 'occasion': 'casual'}
            ],
            'feminine': [
                {'tip': 'Consider a chic cardigan', 'occasion': 'casual'}
            ]
        },
        'formal': {
            'all': [
                {'tip': 'Choose a tailored coat', 'occasion': 'business'}
            ],
            'masculine': [
                {'tip': 'Opt for a classic overcoat', 'occasion': 'formal'}
            ],
            'feminine': [
                {'tip': 'Pair with elegant accessories', 'occasion': 'formal'}
            ]
        }
    },
    'footwear': {
        'casual': {
            'all': [
                {'tip': 'Keep it comfy with stylish sneakers', 'occasion': 'daily'}
            ],
            'masculine': [
                {'tip': 'Match with casual jeans', 'occasion': 'casual'}
            ],
            'feminine': [
                {'tip': 'Try cute flats for a relaxed look', 'occasion': 'casual'}
            ]
        },
        'formal': {
            'all': [
                {'tip': 'Opt for polished shoes', 'occasion': 'business'}
            ],
            'masculine': [
                {'tip': 'Wear dress shoes for a sharp appearance', 'occasion': 'formal'}
            ],
            'feminine': [
                {'tip': 'Pair with elegant heels', 'occasion': 'formal'}
            ]
        }
    },
    'accessories': {
        'casual': {
            'all': [
                {'tip': 'Add a casual watch or bracelet', 'occasion': 'daily'}
            ],
            'masculine': [
                {'tip': 'Try a sleek wallet', 'occasion': 'casual'}
            ],
            'feminine': [
                {'tip': 'Experiment with layered necklaces', 'occasion': 'casual'}
            ]
        },
        'formal': {
            'all': [
                {'tip': 'Keep accessories minimal for elegance', 'occasion': 'business'}
            ],
            'masculine': [
                {'tip': 'Opt for a classic tie or cufflinks', 'occasion': 'formal'}
            ],
            'feminine': [
                {'tip': 'Choose a statement clutch', 'occasion': 'formal'}
            ]
        }
    }
}

def get_style_recommendations(category: str, 
                              confidence: float, 
                              user_preferences: Dict = None,
                              season: str = None) -> List[Dict[str, str]]:
    """Generate personalized style recommendations based on multiple factors."""
    recommendations = []
    base_recommendations = STYLE_RECOMMENDATIONS.get(category, {})
    
    # Get gender preference from user preferences
    gender_style = user_preferences.get('gender_style', 'all')
    
    # Get style type (casual/formal)
    style_type = user_preferences.get('style', 'casual')
    
    # Get recommendations for all genders
    if style_type in base_recommendations:
        recommendations.extend(base_recommendations[style_type].get('all', []))
        # Add gender-specific recommendations
        if gender_style in ['masculine', 'feminine']:
            recommendations.extend(base_recommendations[style_type].get(gender_style, []))
    
    return recommendations

def fetch_api_recommendations(category: str, confidence: float, endpoint: str, api_key: str) -> List[Dict[str, str]]:
    """Simulate fetching additional recommendations from an external API.
       In a real app, you'd use requests or another HTTP library to call the API."""
    if endpoint and api_key:
        # Simulated API response
        return [
            {"tip": "External API suggestion: Try a vintage look!", "occasion": "special"},
            {"tip": "External API suggestion: Consider bold prints for a party vibe!", "occasion": "party"}
        ]
    return []

def display_recommendations(category: str, confidence: float, col, api_endpoint: str, api_key: str, api_enabled: bool):
    """Display recommendations in the Streamlit column."""
    try:
        # Build user preferences from session state
        user_prefs = {
            'style': st.session_state.get('style', 'casual'),
            'gender_style': st.session_state.get('gender_style', 'all'),
            'favorite_colors': ['blue', 'black'],  # Placeholder for future customization
            'occasions': ['work', 'casual']          # Placeholder for future customization
        }
        
        recommendations = get_style_recommendations(
            category=category,
            confidence=confidence,
            user_preferences=user_prefs,
            season=get_current_season()
        )
        
        # If external API is enabled, merge its recommendations
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

@st.cache_resource
def load_model():
    # Load a pre-trained ResNet50 model
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    # Add fashion-specific layers on top of the base model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(FASHION_CATEGORIES), activation='softmax')
    ])
    return model

def preprocess_image(image):
    """Resize and preprocess image for the model."""
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA image
        image = image[:, :, :3]
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

def get_current_season():
    """Determine the current season based on the month."""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

# Initialize session state variables if not already set
if 'gender_style' not in st.session_state:
    st.session_state.gender_style = 'all'
if 'style' not in st.session_state:
    st.session_state.style = 'casual'

# Sidebar for user preferences and advanced settings
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
    # Confidence threshold slider for model predictions
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Set the minimum confidence required to display a prediction."
    )
    
    # External API settings
    api_enabled = st.checkbox("Enable External API Recommendations", value=False)
    api_endpoint = ""
    api_key = ""
    if api_enabled:
        api_endpoint = st.text_input("API Endpoint", value="https://example.com/api")
        api_key = st.text_input("API Key", type="password")
    
    st.write("---")
    st.write("### Additional Settings")
    st.write("Customize more options in future updates!")

# Main content
st.title("👗 FashionFusion: Advanced Fashion Classifier")
st.markdown("Upload an image of your fashion item, and we'll classify it and give you personalized style suggestions!")

# Load the model (with a spinner to keep it cool while loading)
with st.spinner("Loading model... Please wait."):
    model = load_model()

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image and show progress
        with st.spinner('Analyzing your fashion item...'):
            progress_bar = st.progress(0)
            processed_image = preprocess_image(image)
            progress_bar.progress(33)
            
            # Make prediction
            predictions = model.predict(processed_image)
            progress_bar.progress(66)
            
            # Check if any prediction is confidently above threshold
            significant = [conf for conf in predictions[0] if conf > confidence_threshold]
            if not significant:
                st.warning("Hmm... our model isn't too sure about this one. Try a clearer image!")
            progress_bar.progress(100)
        
        # Display classification results and style recommendations side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Results:")
            # Iterate over each category and display if confidence > threshold
            for category, confidence in zip(FASHION_CATEGORIES.keys(), predictions[0]):
                if confidence > confidence_threshold:
                    st.success(f"🏷️ **{category.title()}**: {confidence*100:.1f}%")
                    # For high confidence, list out possible subcategories
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
