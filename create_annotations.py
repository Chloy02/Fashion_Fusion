import os
import json
from pathlib import Path
import random

def create_annotations():
    # Define paths
    base_dir = Path("datasets/deepfashion")
    images_dir = base_dir / "images"
    annotations_file = base_dir / "annotations.json"

    # Check if images directory exists
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return

    # Valid categories from deepfashion_config.json
    categories = [
        "tops",
        "bottoms",
        "dresses",
        "outerwear",
        "footwear",
        "accessories"
    ]

    # Get all image files recursively
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = []
    
    print(f"Scanning directory recursively: {images_dir}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                # Get relative path from images_dir
                rel_path = os.path.relpath(os.path.join(root, file), images_dir)
                image_files.append(rel_path)
    
    print(f"Image files found: {len(image_files)}")
    if len(image_files) == 0:
        print("No image files found with extensions:", valid_extensions)
        return

    # Create annotations
    annotations = {
        "annotations": []
    }

    for image_file in image_files:
        # Randomly assign a category for each image
        category = random.choice(categories)
        
        annotation = {
            "file_name": image_file,
            "category_name": category
        }
        annotations["annotations"].append(annotation)

    # Save annotations to file
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"\nSuccess!")
    print(f"Created annotations for {len(image_files)} images")
    print(f"Annotations saved to: {annotations_file}")

if __name__ == "__main__":
    create_annotations()
