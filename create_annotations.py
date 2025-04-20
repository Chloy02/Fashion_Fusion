import os
import json
from pathlib import Path
import re

def get_category_from_name(name):
    """Determine the main category from the item name"""
    name = name.lower()
    
    # Define category keywords with more specific patterns
    category_keywords = {
        'tops': [
            'tee', 'shirt', 'blouse', 'sweater', 'top', 'tank', 'cami',
            't-shirt', 'tshirt', 'sweatshirt', 'hoodie', 'pullover',
            'sleeve', 'neck', 'collar'
        ],
        'bottoms': [
            'pants', 'jeans', 'shorts', 'skirt', 'leggings',
            'trousers', 'capris', 'culottes', 'palazzo',
            'waist', 'hem', 'pant'
        ],
        'dresses': [
            'dress', 'gown', 'jumper', 'romper', 'jumpsuit',
            'maxi', 'midi', 'mini', 'shift', 'sheath',
            'print', 'floral', 'pattern'
        ],
        'outerwear': [
            'coat', 'jacket', 'blazer', 'cardigan', 'sweater',
            'trench', 'parka', 'vest', 'poncho', 'cape',
            'wrap', 'cover'
        ],
        'accessories': [
            'bag', 'hat', 'scarf', 'belt', 'gloves',
            'sunglasses', 'jewelry', 'watch', 'tie', 'socks',
            'purse', 'wallet', 'backpack', 'headband', 'bracelet',
            'necklace', 'earrings', 'ring', 'brooch', 'pin'
        ]
    }
    
    # Special cases for compound words
    if 'romper' in name or 'jumpsuit' in name:
        return 'dresses'
    if 'cardigan' in name or 'sweater' in name:
        return 'outerwear'
    if 'cami' in name and 'dress' not in name:
        return 'tops'
    
    # Check each category's keywords
    for category, keywords in category_keywords.items():
        if any(keyword in name for keyword in keywords):
            return category
    
    # If no category found, use context clues
    if 'sleeve' in name or 'neck' in name:
        return 'tops'
    if 'waist' in name or 'hem' in name:
        return 'bottoms'
    if 'print' in name or 'floral' in name:
        return 'dresses'
    if 'wrap' in name or 'cover' in name:
        return 'outerwear'
    if 'chain' in name or 'bead' in name or 'crystal' in name:
        return 'accessories'
    
    # Default to 'dresses' if no clear category
    return 'dresses'

def create_annotations():
    # Define paths
    base_dir = Path("datasets/deepfashion")
    images_dir = base_dir / "images"
    annotations_file = base_dir / "annotations.json"

    # Check if images directory exists
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return

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
        # Get the directory name which contains the item description
        dir_name = os.path.dirname(image_file)
        if not dir_name:  # If image is in root directory
            dir_name = os.path.basename(image_file)
        
        # Clean the directory name to get the item name
        item_name = dir_name.replace('_', ' ').lower()
        
        # Determine the category based on the item name
        category = get_category_from_name(item_name)
        
        annotation = {
            "file_name": image_file,
            "category_name": category,
            "item_name": item_name  # Store the full item name for reference
        }
        annotations["annotations"].append(annotation)

    # Save annotations to file
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    # Print statistics
    category_counts = {}
    for ann in annotations["annotations"]:
        cat = ann["category_name"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nAnnotation Statistics:")
    print("====================")
    print(f"Total images annotated: {len(annotations['annotations'])}")
    print("\nImages per category:")
    for cat, count in category_counts.items():
        print(f"{cat}: {count}")

    print(f"\nAnnotations saved to: {annotations_file}")

if __name__ == "__main__":
    create_annotations()
