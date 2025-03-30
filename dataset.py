import tensorflow as tf
import json
import numpy as np
from pathlib import Path

class DeepFashionDataset:
    def __init__(self):
        # Load configs
        with open('config/deepfashion_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load annotations
        with open('datasets/deepfashion/annotations.json', 'r') as f:
            self.annotations = json.load(f)
        
        self.categories = self.config['categories']
        self.num_classes = len(self.categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # Image settings
        self.image_size = self.config['image_size']
        
    def preprocess_image(self, image_path):
        # Read image file
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize and normalize
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = image / 255.0
        return image
        
    def get_batch(self, batch_size):
        # Randomly sample batch_size annotations
        batch_annotations = np.random.choice(self.annotations['annotations'], batch_size)
        
        images = []
        labels = np.zeros((batch_size, self.num_classes))
        
        for i, ann in enumerate(batch_annotations):
            # Construct image path
            image_path = str(Path('datasets/deepfashion/images') / ann['file_name'])
            
            try:
                # Load and preprocess image
                image = self.preprocess_image(image_path)
                images.append(image)
                
                # Create one-hot encoded label
                category_idx = self.category_to_idx[ann['category_name']]
                labels[i, category_idx] = 1
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue
        
        return tf.stack(images), labels