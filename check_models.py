import google.generativeai as genai
import os

# Use the key provided by the user
api_key = "AIzaSyD_phCe9BCP1LS3gvXbz6glEfvyHq7-920"

genai.configure(api_key=api_key)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
