import tensorflow as tf
from app import DeepFashionDataset, load_model
import os
import sys

def train_model():
    try:
        # Initialize dataset
        dataset = DeepFashionDataset()
        
        if not dataset.annotations.get("annotations"):
            print("\nError: DeepFashion dataset not properly configured!")
            print("\nPlease follow these steps:")
            print("1. Download the DeepFashion dataset")
            print("2. Create the following directory structure:")
            print("   datasets/")
            print("   └── deepfashion/")
            print("       ├── annotations.json")
            print("       └── images/")
            print("3. Update config/dataset_config.json with correct paths")
            sys.exit(1)
        
        # Load model
        print("Loading model...")
        model = load_model()
        
        # Training parameters
        EPOCHS = 50
        BATCH_SIZE = 32
        STEPS_PER_EPOCH = 100
        
        # Create directory for checkpoints
        os.makedirs("models", exist_ok=True)
        
        # Training loop
        print("Starting training...")
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            for step in range(STEPS_PER_EPOCH):
                images, labels = dataset.get_batch(BATCH_SIZE)
                loss, accuracy = model.train_on_batch(images, labels)
                if step % 10 == 0:
                    print(f"Step {step}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")
            model.save(f'models/deepfashion_model_epoch_{epoch+1}.h5')
            print(f"Saved model checkpoint for epoch {epoch+1}")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nPlease ensure:")
        print("1. All required dependencies are installed")
        print("2. Dataset is properly configured")
        print("3. Sufficient disk space is available")
        sys.exit(1)

if __name__ == "__main__":
    train_model()
