import os
from groq import Groq
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

class FashionRecommender:
    def __init__(self, model_path='models/final_model.h5'):
        # Initialize Groq client
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.llm_model = "llama-3.3-70b-versatile"
        
        # Load the trained fashion classification model
        self.fashion_model = tf.keras.models.load_model(model_path)
        
        # Categories from your dataset
        self.categories = ['Tops', 'Dresses', 'Bottoms', 'Outerwear', 'Accessories']
        
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def get_item_features(self, image_path):
        """Get features and category prediction for an item"""
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Get model predictions
        predictions = self.fashion_model.predict(img_array)
        category_idx = np.argmax(predictions[0])
        category = self.categories[category_idx]
        confidence = float(predictions[0][category_idx])
        
        # Get features from the last dense layer
        feature_model = tf.keras.Model(
            inputs=self.fashion_model.input,
            outputs=self.fashion_model.get_layer('dense').output
        )
        features = feature_model.predict(img_array)
        
        return {
            'category': category,
            'confidence': confidence,
            'features': features.tolist()[0]
        }
    
    def get_recommendations(self, image_path, user_preferences=None, n_recommendations=5):
        """Generate recommendations based on image and user preferences"""
        # Get item features and category
        item_info = self.get_item_features(image_path)
        
        # Create current item context
        current_item = {
            'category': item_info['category'],
            'confidence': item_info['confidence'],
            'image_path': str(image_path)
        }
        
        # Default user preferences if none provided
        if user_preferences is None:
            user_preferences = {
                'style_preferences': 'versatile, modern',
                'preferred_colors': 'any',
                'occasion': 'casual'
            }
        
        # Create prompt for Groq
        prompt = f"""As a fashion expert, analyze this item and provide {n_recommendations} specific recommendations:

Current Item:
- Category: {current_item['category']} (confidence: {current_item['confidence']:.2f})
- Image Path: {current_item['image_path']}

User Preferences:
- Style: {user_preferences.get('style_preferences', 'versatile')}
- Colors: {user_preferences.get('preferred_colors', 'any')}
- Occasion: {user_preferences.get('occasion', 'casual')}

Based on our fashion classification model's analysis, provide recommendations that would pair well with this item.
Consider the category, style compatibility, and user preferences.

Return the recommendations in this JSON format:
{{
    "recommendations": [
        {{
            "category": "one_of_{','.join(self.categories)}",
            "description": "detailed_description",
            "style_tips": "styling_advice",
            "occasion": "suitable_occasions",
            "confidence_score": "0_to_1_score"
        }}
    ]
}}"""

        try:
            # Get recommendations from Groq
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a professional fashion stylist with expertise in the DeepFashion dataset categories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse and validate recommendations
            recommendations = json.loads(completion.choices[0].message.content)
            
            # Add model's confidence scores
            for rec in recommendations['recommendations']:
                if rec['category'] in self.categories:
                    rec['model_validated'] = True
                else:
                    rec['model_validated'] = False
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return {"error": str(e)}
    
    def get_outfit_compatibility_score(self, image_path1, image_path2):
        """Calculate compatibility score between two items"""
        # Get features for both items
        item1_info = self.get_item_features(image_path1)
        item2_info = self.get_item_features(image_path2)
        
        # Calculate feature similarity
        features1 = np.array(item1_info['features'])
        features2 = np.array(item2_info['features'])
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        
        # Get detailed explanation from Groq
        prompt = f"""Analyze these two fashion items and explain their compatibility:

Item 1: {item1_info['category']} (confidence: {item1_info['confidence']:.2f})
Item 2: {item2_info['category']} (confidence: {item2_info['confidence']:.2f})
Calculated Similarity Score: {similarity:.2f}

Please provide:
1. Style compatibility analysis
2. Whether these categories typically work well together
3. Styling suggestions
4. Confidence in the recommendation (0-1)"""

        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a fashion expert analyzing outfit compatibility."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                'similarity_score': float(similarity),
                'item1_category': item1_info['category'],
                'item2_category': item2_info['category'],
                'explanation': completion.choices[0].message.content
            }
            
        except Exception as e:
            print(f"Error analyzing compatibility: {str(e)}")
            return {"error": str(e)}

    def _create_item_description(self, image_features, category):
        """Convert model features and category into a text description"""
        return f"This is a {category} item with the following style characteristics: {image_features}"
    
    def explain_recommendation(self, current_item, recommended_item):
        """Provide detailed explanation for why items go well together"""
        prompt = f"""Explain in detail why these fashion items complement each other:

Item 1: {current_item['description']}
Item 2: {recommended_item['description']}

Please explain:
1. How the styles complement each other
2. Color coordination
3. Occasion appropriateness
4. Styling tips"""

        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a fashion expert who explains style combinations in detail."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return f"Error: {str(e)}" 