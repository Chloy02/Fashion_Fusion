import tensorflow as tf
from dataset import DeepFashionDataset
import os
import sys
from pathlib import Path
import psutil
import time
from datetime import datetime
import numpy as np

# Set up multi-threading configuration
tf.config.threading.set_inter_op_parallelism_threads(6)  # Half of available cores
tf.config.threading.set_intra_op_parallelism_threads(6)  # Half of available cores

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
    
    # Optimize layer freezing for faster training
    for layer in base_model.layers[:-50]:  # Freeze fewer layers for faster training
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),  # Reduced dense layer size
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a slightly higher learning rate for faster convergence
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
        loss='categorical_crossentropy',
        metrics=['accuracy']
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
    # Setup logging first for proper error handling
    log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("logs", exist_ok=True)
    
    def log_message(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    try:
        dataset = DeepFashionDataset()
        
        # Calculate steps per epoch based on dataset size
        BATCH_SIZE = 64  # Increased batch size
        total_samples = len(dataset.annotations['annotations'])
        train_samples = int(total_samples * 0.8)  # 80% for training
        
        # Limit steps per epoch to make training more manageable
        STEPS_PER_EPOCH = min(500, train_samples // BATCH_SIZE)
        
        log_message(f"Training configuration:")
        log_message(f"Total samples: {total_samples}")
        log_message(f"Training samples: {train_samples}")
        log_message(f"Batch size: {BATCH_SIZE}")
        log_message(f"Steps per epoch: {STEPS_PER_EPOCH}")
        
        # Calculate class weights
        category_counts = {}
        for ann in dataset.annotations['annotations']:
            cat = ann['category_name']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total_samples = sum(category_counts.values())
        class_weights = {}
        for category, count in category_counts.items():
            weight = total_samples / (len(category_counts) * count)
            class_weights[dataset.category_to_idx[category]] = weight
        
        log_message("\nClass weights:")
        for cat, weight in class_weights.items():
            log_message(f"Category {cat}: {weight:.2f}")
        
        model = create_model(len(dataset.categories))
        os.makedirs("models", exist_ok=True)
        
        val_dataset, train_dataset = dataset.get_validation_data(0.2)
        
        # Progress monitoring callback
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.last_batch_time = None
                self.batch_times = []
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                log_message(f"\nStarting epoch {epoch + 1}")
                self.last_batch_time = time.time()
                self.batch_times = []
            
            def on_batch_end(self, batch, logs=None):
                if batch % 10 == 0:  # Update every 10 batches
                    current_time = time.time()
                    if self.last_batch_time:
                        batch_time = current_time - self.last_batch_time
                        self.batch_times.append(batch_time)
                        avg_time = sum(self.batch_times[-5:]) / min(len(self.batch_times), 5)
                        
                        # Calculate ETA for epoch
                        remaining_batches = STEPS_PER_EPOCH - batch
                        eta = remaining_batches * avg_time
                        
                        print(f"\rBatch {batch}/{STEPS_PER_EPOCH} "
                              f"- Loss: {logs['loss']:.4f} "
                              f"- Accuracy: {logs['accuracy']:.4f} "
                              f"- ETA: {eta:.0f}s", end='')
                    
                    self.last_batch_time = current_time
            
            def on_epoch_end(self, epoch, logs=None):
                time_taken = time.time() - self.epoch_start_time
                print()  # New line after batch progress
                log_message(f"Epoch {epoch + 1} completed in {time_taken:.2f} seconds")
                log_message(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
                log_message(f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")
                
                # Monitor system resources
                stats = get_system_stats()
                log_message(f"CPU Usage: {stats['cpu_percent']}%, Memory Usage: {stats['memory_percent']}%")
                if stats['cpu_temp']:
                    log_message(f"CPU Temperature: {stats['cpu_temp']}°C")
        
        # Optimized callbacks
        callbacks = [
            ProgressCallback(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,  # Reduced patience
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/deepfashion_model_epoch_{epoch:02d}.h5',
                save_best_only=True,
                monitor='val_accuracy',
                save_weights_only=False
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Start training with proper steps
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=15,  # Reduced epochs
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=STEPS_PER_EPOCH // 5,  # 20% of steps for validation
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0  # Using custom progress monitoring
        )
        
        model.save('models/final_model.h5')
        log_message("\nTraining completed successfully!")
        log_message(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
    except Exception as e:
        log_message(f"\nError during training: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        sys.exit(1)

def test_dataset():
    dataset = DeepFashionDataset()
    
    # Get validation data
    val_dataset, train_dataset = dataset.get_validation_data(0.2)
    
    # Test training batch
    train_batch = next(iter(train_dataset))
    print("\nTraining batch test:")
    print(f"Images shape: {train_batch[0].shape}")
    print(f"Labels shape: {train_batch[1].shape}")
    # Convert tensor to numpy for min/max calculation
    images_np = train_batch[0].numpy()
    print(f"Image value range: [{images_np.min():.2f}, {images_np.max():.2f}]")
    
    # Test validation data
    print("\nValidation data test:")
    val_batch = next(iter(val_dataset))
    print(f"Validation images shape: {val_batch[0].shape}")
    print(f"Validation labels shape: {val_batch[1].shape}")

if __name__ == "__main__":
    test_dataset()
    # If tests pass, proceed with training
    train_model()
