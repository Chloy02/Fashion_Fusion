import tensorflow as tf
from dataset import DeepFashionDataset
import os
import sys
from pathlib import Path
import psutil
import time
from datetime import datetime

def get_system_stats():
    cpu_temp = psutil.sensors_temperatures().get('coretemp', [])
    cpu_temp = max([temp.current for temp in cpu_temp]) if cpu_temp else None
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_temp': cpu_temp
    }

def should_pause_training(stats, temp_threshold=85, cpu_threshold=95, memory_threshold=90):
    """Adjusted thresholds for laptop CPU"""
    if stats['cpu_temp'] and stats['cpu_temp'] > temp_threshold:
        return True, "CPU temperature too high"
    if stats['cpu_percent'] > cpu_threshold:
        return True, "CPU usage too high"
    if stats['memory_percent'] > memory_threshold:
        return True, "Memory usage too high"
    return False, None

def create_model(num_classes):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Fine-tune the last few layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model

def find_latest_checkpoint():
    checkpoint_dir = Path("models")
    if not checkpoint_dir.exists():
        return None, 0
    
    # Updated to look for .keras files instead of .h5
    checkpoints = list(checkpoint_dir.glob("deepfashion_model_epoch_*.keras"))
    if not checkpoints:
        return None, 0
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    epoch_number = int(latest_checkpoint.stem.split('_')[-1])
    return latest_checkpoint, epoch_number

def train_model():
    try:
        # Initialize dataset
        dataset = DeepFashionDataset()
        
        # Training parameters
        TOTAL_EPOCHS = 100  # Increase epochs
        BATCH_SIZE = 32
        STEPS_PER_EPOCH = len(dataset.annotations['annotations']) // BATCH_SIZE
        
        # Learning rate schedule with warmup
        initial_learning_rate = 0.0001
        warmup_epochs = 5
        decay_epochs = 20
        
        def learning_rate_schedule(epoch):
            if epoch < warmup_epochs:
                return initial_learning_rate * ((epoch + 1) / warmup_epochs)
            else:
                return initial_learning_rate * (0.1 ** ((epoch - warmup_epochs) // decay_epochs))
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)
        
        # System monitoring parameters
        TEMP_THRESHOLD = 85  # Ryzen mobile CPUs should stay under 75°C
        COOLING_PAUSE = 180  # 3 minutes cooling pause
        
        # Create directory for checkpoints and logs
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Setup logging
        log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        def log_message(message):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            print(log_entry)
            with open(log_file, 'a') as f:
                f.write(log_entry + '\n')

        # Find latest checkpoint
        latest_checkpoint, start_epoch = find_latest_checkpoint()
        
        # Create or load model
        if latest_checkpoint is not None:
            log_message(f"Resuming from checkpoint: {latest_checkpoint} (Epoch {start_epoch})")
            model = tf.keras.models.load_model(latest_checkpoint)
        else:
            log_message("Starting fresh training...")
            model = create_model(len(dataset.categories))
            start_epoch = 0
        
        # Update optimizer with learning rate schedule
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Add validation data if available
        validation_data = dataset.get_validation_data()  # You'll need to implement this
        
        # Training loop
        training_start_time = time.time()
        
        try:
            for epoch in range(start_epoch, TOTAL_EPOCHS):
                epoch_start_time = time.time()
                log_message(f"\nStarting Epoch {epoch + 1}/{TOTAL_EPOCHS}")
                
                for step in range(STEPS_PER_EPOCH):
                    # Check system stats every 10 steps
                    if step % 10 == 0:
                        stats = get_system_stats()
                        should_pause, reason = should_pause_training(stats)
                        
                        if should_pause:
                            log_message(f"\nPausing training: {reason}")
                            log_message(f"CPU: {stats['cpu_percent']}%, Memory: {stats['memory_percent']}%, Temperature: {stats['cpu_temp']}°C")
                            log_message(f"Saving checkpoint before cooling pause...")
                            
                            # Save temporary checkpoint in new format
                            temp_checkpoint = f'models/temp_checkpoint_epoch_{epoch+1}_step_{step}.keras'
                            model.save(temp_checkpoint)  # Remove save_format parameter
                            
                            log_message(f"Cooling pause for {COOLING_PAUSE} seconds...")
                            time.sleep(COOLING_PAUSE)
                            
                            log_message("Resuming training...")
                            
                    # Training step
                    images, labels = dataset.get_batch(BATCH_SIZE)
                    loss, accuracy = model.train_on_batch(images, labels)
                    
                    if step % 10 == 0:
                        elapsed_time = time.time() - training_start_time
                        log_message(f"Step {step}: loss = {loss:.4f}, accuracy = {accuracy:.4f} (Total training time: {elapsed_time/3600:.1f}h)")
                
                # Save epoch checkpoint in new format
                checkpoint_path = f'models/deepfashion_model_epoch_{epoch+1}.keras'
                model.save(checkpoint_path)  # Remove save_format parameter
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                log_message(f"\nEpoch {epoch + 1} completed in {epoch_time/60:.1f} minutes")
                log_message(f"Checkpoint saved: {checkpoint_path}")
                
                # Save latest version in new format
                model.save('models/latest_model.keras')  # Remove save_format parameter
                
                # Add validation step at end of each epoch
                if validation_data:
                    val_loss, val_accuracy = model.evaluate(validation_data[0], validation_data[1])
                    log_message(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        except KeyboardInterrupt:
            log_message("\nTraining interrupted by user...")
            interrupt_checkpoint = f'models/interrupted_epoch_{epoch+1}_step_{step}.keras'
            model.save(interrupt_checkpoint)  # Remove save_format parameter
            log_message(f"Saved interrupt checkpoint: {interrupt_checkpoint}")
            sys.exit(0)

    except Exception as e:
        log_message(f"\nError during training: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        sys.exit(1)

def test_dataset():
    dataset = DeepFashionDataset()
    
    # Test single batch
    images, labels = dataset.get_batch(4)
    print("\nBatch test:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")
    
    # Test validation data
    val_images, val_labels = dataset.get_validation_data(0.2)
    print("\nValidation data test:")
    print(f"Validation images shape: {val_images.shape}")
    print(f"Validation labels shape: {val_labels.shape}")

if __name__ == "__main__":
    test_dataset()
    # If tests pass, proceed with training
    train_model()
