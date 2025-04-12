import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import sys
import os

# Configuration
MODEL_PATH = r"C:\Users\syedn\Documents\DeepfakeP\best_model.keras"
IMAGE_SIZE = 224
THRESHOLD = 0.5  # Reset to default threshold
TTA_ITERATIONS = 5  # Number of test-time augmentations

def load_and_preprocess_image(image_path):
    """Load and preprocess the image with ResNet50 specific preprocessing"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Proper ResNet50 preprocessing
    return np.expand_dims(img_array, axis=0)

def predict_with_tta(model, image_path):
    """Make prediction using test-time augmentation"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    x = np.array(img)
    x = preprocess_input(x)
    
    # Create augmented versions
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    preds = []
    for i, augmented in enumerate(datagen.flow(np.expand_dims(x, 0), batch_size=1)):
        preds.append(model.predict(augmented, verbose=0)[0][0])
        if i >= TTA_ITERATIONS - 1:
            break
    
    return np.mean(preds)  # Return average prediction

def predict(image_path):
    """Predict if the image is real or fake with confidence score"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    try:
        model = load_model(MODEL_PATH)
        
        # First do a simple prediction for debugging
        img_array = load_and_preprocess_image(image_path)
        debug_pred = model.predict(img_array, verbose=0)[0][0]
        print(f"\n[Debug] Raw initial prediction: {debug_pred:.4f}")
        
        # Choose prediction method
        use_tta = True  # Set to False for faster single prediction
        
        if use_tta:
            confidence = predict_with_tta(model, image_path)
            print(f"[Debug] TTA-adjusted confidence: {confidence:.4f}")
        else:
            confidence = debug_pred
        
        # Apply threshold with balanced interpretation
        if confidence > THRESHOLD:
            prediction = "Fake"
            confidence_pct = confidence * 100
        else:
            prediction = "Real"
            confidence_pct = (1 - confidence) * 100
        
        # Print detailed results
        print("\n=== Final Prediction ===")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence_pct:.2f}%")
        print(f"Threshold: {THRESHOLD}")
        print(f"Raw score: {confidence:.4f} (0=Real, 1=Fake)")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py test_image.jpg")
    else:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
        else:
            predict(image_path)