import os
from recommendation_service import FashionRecommender
from pathlib import Path

def test_recommendations():
    # Initialize the recommender with your trained model
    recommender = FashionRecommender(model_path='models/final_model.h5')
    
    # Example image path - replace with an actual image from your dataset
    test_image_path = "path/to/your/test/image.jpg"  # You'll need to provide a real image path
    
    # Example user preferences
    user_preferences = {
        'style_preferences': 'casual, modern',
        'preferred_colors': ['black', 'white', 'navy'],
        'occasion': 'casual'
    }
    
    print("1. Testing single item recommendations:")
    print("-" * 50)
    
    # Get recommendations for a single item
    recommendations = recommender.get_recommendations(
        image_path=test_image_path,
        user_preferences=user_preferences,
        n_recommendations=3
    )
    
    if "recommendations" in recommendations:
        print("\nRecommended items:")
        for i, rec in enumerate(recommendations["recommendations"], 1):
            print(f"\nRecommendation {i}:")
            print(f"Category: {rec['category']}")
            print(f"Description: {rec['description']}")
            print(f"Style Tips: {rec['style_tips']}")
            print(f"Occasion: {rec['occasion']}")
            print(f"Confidence Score: {rec['confidence_score']}")
            print(f"Model Validated: {rec['model_validated']}")
    else:
        print("Error getting recommendations:", recommendations.get("error"))
    
    print("\n2. Testing outfit compatibility:")
    print("-" * 50)
    
    # Example: Test compatibility between two items
    second_image_path = "path/to/your/second/test/image.jpg"  # You'll need to provide a real image path
    
    compatibility = recommender.get_outfit_compatibility_score(
        test_image_path,
        second_image_path
    )
    
    if "error" not in compatibility:
        print(f"\nCompatibility Results:")
        print(f"Similarity Score: {compatibility['similarity_score']:.2f}")
        print(f"Item 1 Category: {compatibility['item1_category']}")
        print(f"Item 2 Category: {compatibility['item2_category']}")
        print("\nDetailed Analysis:")
        print(compatibility['explanation'])
    else:
        print("Error analyzing compatibility:", compatibility.get("error"))

if __name__ == "__main__":
    # Make sure GROQ_API_KEY is set
    if not os.getenv('GROQ_API_KEY'):
        print("Please set your GROQ_API_KEY environment variable")
        exit(1)
    
    test_recommendations() 