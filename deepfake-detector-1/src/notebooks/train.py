import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

# === PARAMETERS ===
image_size = 224
batch_size = 32
initial_epochs = 5
fine_tune_epochs = 15
num_classes = 1  # Binary classification

# Paths (update these)
train_data_dir = r"C:\Users\syedn\Documents\DeepfakeP\deepfake-detector-1\src\data\train"
val_data_dir = r"C:\Users\syedn\Documents\DeepfakeP\deepfake-detector-1\src\data\val"

# === DATA PREPARATION ===
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Create FINAL train generator first
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=['fake', 'real'],  # Set class order FIRST
    shuffle=True
)

# Calculate class weights AFTER setting classes
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator)
class_weights = {
    0: total_samples / (2 * class_counts[0]),  # fake
    1: total_samples / (2 * class_counts[1])   # real
}

# Validation generator with same class order
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary',
    classes=['fake', 'real']  # Match training order
)

# === MODEL ARCHITECTURE ===
def build_model(image_size):
    base_model = ResNet50(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze initially

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)

model = build_model(image_size)

# === TRAINING CONFIGURATION ===
initial_lr = 1e-4
model.compile(optimizer=tf.keras.optimizers.Adam(initial_lr),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === INITIAL TRAINING (FROZEN BASE) ===
print("Training initial model with frozen base...")
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=val_generator,
    class_weight=class_weights
)

# === FINE-TUNING (UNFREEZE TOP LAYERS) ===
base_model = model.layers[1]
base_model.trainable = True

# Unfreeze last 20 layers of base model
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Lower learning rate for fine-tuning
fine_tune_lr = initial_lr / 10
model.compile(optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Add callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

print("\nFine-tuning last layers...")
total_history = model.fit(
    train_generator,
    epochs=initial_epochs + fine_tune_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# === SAVE FINAL MODEL ===
model.save('deepfake_detector.keras')
print("✅ Model saved")

# === VISUALIZATION ===
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + total_history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'] + total_history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.axvline(initial_epochs-1, color='gray', linestyle='--', label='Fine-tuning Start')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + total_history.history['loss'], label='Train')
plt.plot(history.history['val_loss'] + total_history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.axvline(initial_epochs-1, color='gray', linestyle='--', label='Fine-tuning Start')
plt.legend()

plt.tight_layout()
plt.show()