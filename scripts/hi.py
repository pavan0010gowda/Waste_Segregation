import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- CONFIGURATION ---
# CRITICAL: This list must match the order printed by train_mobilenet.py
labels = ['dry_decomposable', 'dry_non_degradable', 'dry_recyclable', 'wet']

# Path to your NEW MobileNet model
model_path = '../models/waste_classifier_mobilenet.h5' 
# ---------------------

print(f"Loading model from {model_path}...")
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except IOError:
    print("❌ ERROR: Model file not found!")
    print(f"   Make sure you ran 'train_mobilenet.py' first and the file exists at: {model_path}")
    exit()

# Initialize Camera
# Try index 0 first. If you use an external webcam, change this to 1.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Could not open camera.")
    exit()

print("✅ Camera started. Point at waste items! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- PREPROCESSING FOR MOBILENETV2 ---
    
    # 1. Resize: MobileNetV2 expects 224x224
    img = cv2.resize(frame, (224, 224))
    
    # 2. Color Conversion: OpenCV is BGR, Model needs RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize: Scale pixel values to 0-1
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- PREDICTION ---
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    
    class_idx = np.argmax(score)
    confidence = score[class_idx]
    label_name = labels[class_idx]

    # --- VISUALIZATION ---
    # Green if confident (> 70%), Red if unsure
    if confidence > 0.7:
        color = (0, 255, 0) # Green
        status_text = f"{label_name}: {confidence*100:.0f}%"
    else:
        color = (0, 0, 255) # Red
        status_text = f"Unsure: {confidence*100:.0f}%"
    
    # Text Settings: (X, Y), Font, Size (1.2), Color, Thickness (3)
    cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow('Waste Detection System', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()