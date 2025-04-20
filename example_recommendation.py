from recommendation_service import FashionRecommender

def main():
    # Initialize the recommender
    recommender = FashionRecommender()
    
    # Example user preferences
    user_preferences = {
        "style_preferences": "casual, modern, minimalist",
        "size": "M",
        "preferred_colors": ["black", "white", "navy", "gray"]
    }
    
    # Example current item
    current_item = {
        "category": "Tops",
        "style": "casual t-shirt",
        "description": "A plain white cotton t-shirt with a crew neck",
        "color": "white"
    }
    
    # Get recommendations
    print("Generating outfit recommendations...")
    recommendations = recommender.get_recommendations(user_preferences, current_item)
    
    # Print recommendations
    if "recommendations" in recommendations:
        print("\nRecommended items to pair with your white t-shirt:")
        for i, rec in enumerate(recommendations["recommendations"], 1):
            print(f"\nRecommendation {i}:")
            print(f"Category: {rec['category']}")
            print(f"Description: {rec['description']}")
            print(f"Style: {rec['style']}")
            print(f"Reason: {rec['reason']}")
            
            # Get detailed explanation
            print("\nDetailed styling explanation:")
            explanation = recommender.explain_recommendation(current_item, rec)
            print(explanation)
            print("-" * 80)
    else:
        print("Error getting recommendations:", recommendations.get("error"))

if __name__ == "__main__":
    main() 