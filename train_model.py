import tensorflow as tf
from dataset import DeepFashionDataset
import os
import sys

def create_model(num_classes):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    try:
        # Initialize dataset
        dataset = DeepFashionDataset()
        
        # Create model
        print("Creating model...")
        model = create_model(len(dataset.categories))
        
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
            
            # Save model checkpoint
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
