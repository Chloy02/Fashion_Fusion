import torch
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
from PIL import Image
import numpy as np
import os

print("Testing imports...")
try:
    import streamlit
    print("Streamlit imported.")
except ImportError:
    print("Streamlit import failed!")

print("Loading CLIP model (this might take a moment)...")
try:
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    print("CLIP model loaded successfully.")
except Exception as e:
    print(f"Failed to load CLIP model: {e}")

print("Testing dummy classification...")
try:
    # Create a dummy image (black square)
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    categories = ["object"]
    inputs = processor(text=categories, images=dummy_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    print("Dummy classification successful.")
except Exception as e:
    print(f"Dummy classification failed: {e}")

print("Verification complete.")
