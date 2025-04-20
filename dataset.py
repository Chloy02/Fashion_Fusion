import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import json
import numpy as np
import os
import random
from pathlib import Path

class DeepFashionDataset:
    def __init__(self, config_path='config/deepfashion_config.json'):
        self.config = self._load_config(config_path)
        self.annotations = self._load_annotations()
        self.categories = list(self.config['categories'].keys())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.image_dir = 'datasets/deepfashion/images'
        
        # Print dataset statistics
        self._print_dataset_stats()
        
        # Create TensorFlow dataset with optimized parallel processing
        self._create_data_pipeline()
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {config_path}")
            raise
    
    def _load_annotations(self):
        """Load annotations from JSON file"""
        try:
            with open('datasets/deepfashion/annotations.json', 'r') as f:
                annotations = json.load(f)
            if not self._verify_annotations(annotations):
                raise ValueError("Invalid annotations format")
            return annotations
        except FileNotFoundError:
            print("Error: annotations.json not found in datasets/deepfashion/")
            raise
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in annotations.json")
            raise
    
    def _verify_annotations(self, annotations):
        """Verify the format of annotations"""
        if not isinstance(annotations, dict) or 'annotations' not in annotations:
            return False
        for ann in annotations['annotations']:
            if not all(key in ann for key in ['file_name', 'category_name']):
                return False
            if ann['category_name'] not in self.config['categories']:
                return False
        return True
    
    def _create_data_pipeline(self):
        """Create TensorFlow dataset pipeline with optimized parallel processing"""
        # Convert lists to numpy arrays first for better tensor conversion
        image_paths = [os.path.join(self.image_dir, ann['file_name']) 
                      for ann in self.annotations['annotations']]
        category_indices = [self.category_to_idx[ann['category_name']] 
                           for ann in self.annotations['annotations']]
        
        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices({
            'image_path': image_paths,
            'category_idx': category_indices
        })
        
        # Optimize shuffling for memory and speed
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        
        # Optimize parallel processing for image loading
        dataset = dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Optimize batching and prefetching
        dataset = dataset.batch(64)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        self.dataset = dataset
    
    def _load_and_preprocess_image(self, x):
        """Load and preprocess a single image with optimized caching"""
        # Read and decode image
        image = tf.io.read_file(x['image_path'])
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize with parallel processing optimization
        image = tf.image.resize(image, [224, 224], method='bilinear')
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        
        # Create one-hot label
        label = tf.one_hot(x['category_idx'], depth=len(self.categories))
        
        return image, label
    
    def get_validation_data(self, validation_split=0.2):
        """Split dataset into training and validation sets with optimized processing"""
        total_samples = len(self.annotations['annotations'])
        val_size = int(total_samples * validation_split)
        
        # Create paths and indices lists
        all_image_paths = [os.path.join(self.image_dir, ann['file_name']) 
                          for ann in self.annotations['annotations']]
        all_category_indices = [self.category_to_idx[ann['category_name']] 
                              for ann in self.annotations['annotations']]
        
        # Create validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices({
            'image_path': all_image_paths[:val_size],
            'category_idx': all_category_indices[:val_size]
        })
        
        # Create training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices({
            'image_path': all_image_paths[val_size:],
            'category_idx': all_category_indices[val_size:]
        })
        
        # Optimize validation dataset pipeline
        val_dataset = val_dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(64).prefetch(tf.data.AUTOTUNE)
        
        # Optimize training dataset pipeline
        train_dataset = train_dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(buffer_size=2000)
        train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
        
        return val_dataset, train_dataset
    
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
