import streamlit as st
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the Streamlit page config
st.set_page_config(page_title="FashionFusion", layout="wide", page_icon="👗")

# --- Constants & Configuration ---
FASHION_CATEGORIES = [
    "t-shirt", "shirt", "blouse", "sweater", "hoodie", "jacket", "coat", "blazer",
    "jeans", "trousers", "shorts", "skirt", "dress", "jumpsuit",
    "sneakers", "boots", "heels", "sandals", "loafers",
    "bag", "hat", "scarf", "sunglasses"
]

# --- Model Loading ---
@st.cache_resource
def load_clip_model():
    """Load the CLIP model and processor."""
    # Use a larger model for better accuracy
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

# --- Core Logic ---
def classify_image(image, model, processor, categories):
    """Classify the image using CLIP zero-shot classification."""
    # Prepare text descriptions for CLIP (improves accuracy)
    text_inputs = [f"a photo of a {cat}, a type of clothing" for cat in categories]
    
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Get top predictions
    values, indices = probs[0].topk(5)
    results = []
    for value, index in zip(values, indices):
        results.append((categories[index], value.item()))
    
    return results

def get_dynamic_suggestions(item, occasion, gender_style, season, api_key):
    """Generate style suggestions using Google Gemini API."""
    if not api_key:
        return ["Please enter a valid Google Gemini API Key to get personalized suggestions."]
    
    try:
        genai.configure(api_key=api_key)
        # Use gemini-2.0-flash as requested/verified
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Act as a high-end fashion stylist. I have a {item}.
        Context:
        - Occasion: {occasion}
        - Style Preference: {gender_style}
        - Season: {season}
        
        Please provide 3-4 specific, actionable, and stylish outfit recommendations incorporating this {item}. 
        Focus on color coordination, layering, and accessories. 
        Keep the tone professional yet engaging. 
        Format the output as a bulleted list.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return [f"Error generating suggestions: {str(e)}"]

# --- UI Components ---
def main():
    st.title("👗 FashionFusion: AI Stylist")
    st.markdown("### Upload your fashion item and let AI curate your look.")

    # Sidebar for Settings
    with st.sidebar:
        st.header("Settings")
        # Load key from environment variable, but allow user override
        env_key = os.getenv("GOOGLE_API_KEY", "")
        api_key = st.text_input("Google Gemini API Key", value=env_key, type="password", help="Required for AI suggestions")
        
        st.divider()
        
        st.subheader("Preferences")
        gender_style = st.selectbox("Style Preference", ["Neutral", "Feminine", "Masculine"])
        occasion = st.selectbox("Occasion", ["Casual", "Work/Business", "Party/Event", "Date Night", "Gym/Sport"])
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        
        st.info("Tip: Get your API key from Google AI Studio.")

    # Main Area
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Your Item", use_container_width=True)
        
        with col2:
            with st.spinner("Analyzing style..."):
                # Load Model
                model, processor = load_clip_model()
                
                # Classify
                predictions = classify_image(image, model, processor, FASHION_CATEGORIES)
                top_item, top_conf = predictions[0]
                
                st.subheader("I see a...")
                st.markdown(f"### **{top_item.title()}**")
                st.progress(top_conf)
                st.caption(f"Confidence: {top_conf:.1%}")
                
                with st.expander("See other possibilities"):
                    for item, conf in predictions[1:]:
                        st.write(f"{item.title()}: {conf:.1%}")

            st.divider()
            
            if st.button("✨ Generate Style Suggestions", type="primary"):
                with st.spinner("Consulting the AI stylist..."):
                    suggestions = get_dynamic_suggestions(top_item, occasion, gender_style, season, api_key)
                    
                    st.subheader("Stylist Recommendations")
                    if isinstance(suggestions, list):
                        for s in suggestions:
                            st.write(s)
                    else:
                        st.markdown(suggestions)

if __name__ == "__main__":
    main()