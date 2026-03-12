import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
# CRITICAL: This list must match the order printed by train_mobilenet.py
labels = ['dry_decomposable', 'dry_non_degradable', 'dry_recyclable', 'wet']

# Path to model and training data
model_path = '../models/waste_classifier_mobilenet.h5' 
train_dir = '../data/train' # We need this to save new images
# ---------------------

print(f"Loading model from {model_path}...")
try:
    model = load_model(model_path)
    
    # CRITICAL FIX 1: Re-compile with 'run_eagerly=True' to allow single-image training
    # This prevents the "numpy() is only available when eager execution is enabled" error.
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'], 
                  run_eagerly=True)
                  
    print("✅ Model loaded and compiled for Spot Training.")
except IOError:
    print("❌ ERROR: Model file not found!")
    print(f"   Make sure you ran 'train_mobilenet.py' first.")
    exit()

# Start Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Could not open camera.")
    exit()

print("\n" + "="*50)
print("✅ SYSTEM READY")
print("👉 Point at object.")
print("👉 Press 'x' to CORRECT the model (Train on Spot).")
print("👉 Press 'q' to Quit.")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. PREPARE IMAGE ---
    # Resize and convert for Model
    img_resized = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize (0-1) for prediction
    img_array = np.array(img_rgb) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- 2. PREDICT ---
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    class_idx = np.argmax(score)
    confidence = score[class_idx]
    label_name = labels[class_idx]

    # --- 3. VISUALIZE ---
    if confidence > 0.7:
        color = (0, 255, 0) # Green
        status_text = f"{label_name}: {confidence*100:.0f}%"
    else:
        color = (0, 0, 255) # Red
        status_text = f"Unsure: {confidence*100:.0f}%"
    
    cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow('Waste Detection (Press "x" to Correct)', frame)

    # --- 4. CONTROLS ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # === CORRECTION MODE (Press 'x') ===
    if key == ord('x'):
        print("\n⏸️  SYSTEM PAUSED FOR CORRECTION")
        
        # 1. Ask User for Correct Label
        print("What is this object actually?")
        for i, label in enumerate(labels):
            print(f"  [{i}] {label}")
        
        try:
            user_input = input("Enter number (or press Enter to cancel): ")
            if user_input.strip() == "":
                print("Correction cancelled.")
                continue

            correct_idx = int(user_input)
            
            if 0 <= correct_idx < len(labels):
                correct_label = labels[correct_idx]
                print(f"✅ You said this is: {correct_label}")

                # 2. Save Image to Dataset (So it stays learned forever)
                save_path = os.path.join(train_dir, correct_label)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                timestamp = int(time.time())
                filename = f"manual_correction_{timestamp}.jpg"
                full_save_path = os.path.join(save_path, filename)
                
                # Save the original frame (BGR format for OpenCV)
                cv2.imwrite(full_save_path, frame)
                print(f"💾 Image saved to: {full_save_path}")

                # 3. TRAIN ON SPOT (Incremental Learning)
                print("🧠 Retraining model on this image... ", end="")
                
                # Create the target vector (e.g., [0, 0, 1, 0] for class 2)
                y_true = np.zeros((1, len(labels)))
                y_true[0, correct_idx] = 1.0

                # CRITICAL FIX 2: Use train_on_batch instead of fit
                # This is faster and works better for single images
                loss, acc = model.train_on_batch(img_array, y_true)
                
                print("DONE!")
                print(f"   (New Loss: {loss:.4f})")
                print("Resume detecting...")

            else:
                print("❌ Invalid number selected.")
        
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

cap.release()
cv2.destroyAllWindows()