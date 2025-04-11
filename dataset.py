import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import numpy as np
import os
import random
from pathlib import Path

class DeepFashionDataset:
    def __init__(self):
        # Load configs first
        with open('config/deepfashion_config.json', 'r') as f:
            self.config = json.load(f)
            
        # Set categories before verification
        self.categories = self.config['categories']
        self.num_classes = len(self.categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Image settings for ResNet50
        self.image_size = (224, 224)
        self.is_training = True
        
        # Load annotations
        try:
            with open('datasets/deepfashion/annotations.json', 'r') as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            print("Error: annotations.json not found in datasets/deepfashion/")
            print("Please run create_annotations.py first")
            raise
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in annotations.json")
            raise
            
        # Verify annotations format
        if not self._verify_annotations():
            raise ValueError("Invalid annotations format. Please check the structure of annotations.json")
        
        # Print dataset statistics
        self._print_dataset_stats()
    
    def _verify_annotations(self):
        """Verify annotations format and image files existence"""
        try:
            # Check basic structure
            if 'annotations' not in self.annotations:
                print("Error: Missing 'annotations' key in annotations file")
                return False
            
            # Check each annotation
            for ann in self.annotations['annotations']:
                if not all(key in ann for key in ['file_name', 'category_name']):
                    print(f"Error: Missing required keys in annotation: {ann}")
                    return False
                
                # Check if image file exists
                img_path = Path('datasets/deepfashion/images') / ann['file_name']
                if not img_path.exists():
                    print(f"Error: Image file not found: {img_path}")
                    return False
                
                # Verify category exists in config
                if ann['category_name'] not in self.categories:
                    print(f"Error: Invalid category '{ann['category_name']}' not found in config")
                    print(f"Valid categories are: {list(self.categories)}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error verifying annotations: {str(e)}")
            return False
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        category_counts = {}
        for ann in self.annotations['annotations']:
            cat = ann['category_name']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
        print("\nDataset Statistics:")
        print(f"Total images: {len(self.annotations['annotations'])}")
        print(f"Number of categories: {len(self.categories)}")
        print("\nImages per category:")
        for cat, count in category_counts.items():
            print(f"{cat}: {count}")
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess image for ResNet50"""
        try:
            # Read image file
            img = tf.io.read_file(image_path)
            # Decode image
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            # Convert to float32
            img = tf.cast(img, tf.float32)
            # Resize with padding to maintain aspect ratio
            img = tf.image.resize_with_pad(img, self.image_size[0], self.image_size[1])
            # Preprocess for ResNet50
            img = preprocess_input(img)
            
            return img
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def _apply_augmentation(self, image):
        """Apply augmentation during training"""
        if self.is_training:
            # Random flip
            image = tf.image.random_flip_left_right(image)
            
            # Random brightness (adjusted for ResNet preprocessed images)
            image = tf.image.random_brightness(image, 0.2)
            
            # Random contrast (adjusted for ResNet preprocessed images)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            
            # Random rotation
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            
        return image
    
    def get_batch(self, batch_size):
        """Get a batch of images and labels"""
        images = []
        labels = np.zeros((batch_size, self.num_classes))
        
        while len(images) < batch_size:
            ann = random.choice(self.annotations['annotations'])
            img_path = os.path.join('datasets/deepfashion/images', ann['file_name'])
            
            # Load and preprocess image
            img = self._load_and_preprocess_image(img_path)
            if img is None:
                continue
                
            # Apply augmentation
            img = self._apply_augmentation(img)
            
            # Get label
            category = ann['category_name']
            label_idx = self.category_to_idx[category]
            
            images.append(img)
            labels[len(images)-1, label_idx] = 1
            
        return np.array(images), labels
    
    def get_validation_data(self, validation_split=0.2):
        """Create validation dataset"""
        # Temporarily disable training augmentations
        self.is_training = False
        
        # Randomly select validation samples
        total_samples = len(self.annotations['annotations'])
        val_size = int(total_samples * validation_split)
        val_indices = random.sample(range(total_samples), val_size)
        
        val_images = []
        val_labels = np.zeros((val_size, self.num_classes))
        
        for idx, i in enumerate(val_indices):
            ann = self.annotations['annotations'][i]
            img_path = os.path.join('datasets/deepfashion/images', ann['file_name'])
            
            # Load and preprocess image
            img = self._load_and_preprocess_image(img_path)
            if img is not None:
                val_images.append(img)
                label_idx = self.category_to_idx[ann['category_name']]
                val_labels[idx, label_idx] = 1
        
        self.is_training = True
        return np.array(val_images), val_labels
