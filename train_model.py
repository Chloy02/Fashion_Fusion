import tensorflow as tf
from app import DeepFashionDataset, load_model
import os

def train_model():
    # Initialize the dataset
    dataset = DeepFashionDataset()
    
    # Load the pre-configured model
    model = load_model()
    
    # Training parameters
    EPOCHS = 50
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 100
    
    # Create directory to save model checkpoints if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for step in range(STEPS_PER_EPOCH):
            images, labels = dataset.get_batch(BATCH_SIZE)
            loss, accuracy = model.train_on_batch(images, labels)
            if step % 10 == 0:
                print(f"Step {step}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")
        model.save(f'models/deepfashion_model_epoch_{epoch+1}.h5')
        print(f"Saved model checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    train_model()
