import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- CONFIGURATION ---
train_dir = "data/train"
valid_dir = "data/test"
model_save_path = "../models/waste_classifier.h5"
# ---------------------

# 1. Setup Generators with Augmentation
# We use this to create "fake" variations (zoom, rotate) so the model learns better
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_gen = ImageDataGenerator(rescale=1./255)

# 2. Load Data
print("Loading data...")
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

valid_data = valid_gen.flow_from_directory(
    valid_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# CRITICAL: Print the class mapping. You MUST update live_detect.py with this order.
print("\n" + "="*40)
print(f"✅ DETECTED CLASSES: {list(train_data.class_indices.keys())}")
print(f"✅ MAPPING: {train_data.class_indices}")
print("="*40 + "\n")

# 3. Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Helps prevent memorization
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train
print("Starting training...")
model.fit(train_data, epochs=25, validation_data=valid_data)

# 5. Save
model.save(model_save_path)
print(f"Model saved successfully to {model_save_path}")