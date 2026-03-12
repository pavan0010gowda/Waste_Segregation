# train_mobilenet.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. Setup Data Generators (Same as before)
train_dir = "data/train"
valid_dir = "data/test"

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,  # Increased shift
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224), # MobileNet expects 224x224
    batch_size=32,
    class_mode='categorical'
)

valid_data = valid_gen.flow_from_directory(
    valid_dir,
    target_size=(224, 224), # Match input size
    batch_size=32,
    class_mode='categorical'
)

# Print labels for your reference
print(f"✅ Classes: {train_data.class_indices}")

# 2. Load MobileNetV2 (Pre-trained on ImageNet)
# include_top=False removes the last layer (which was 1000 classes)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model (so we don't ruin its pre-trained knowledge)
base_model.trainable = False 

# 3. Add Custom Layers for Waste Classification
x = base_model.output
x = GlobalAveragePooling2D()(x) # Better than Flatten() for this model
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile
# We use a lower learning rate (0.0001) for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Train
print("Starting training with MobileNetV2...")
model.fit(train_data, epochs=10, validation_data=valid_data)

# 6. Save
model.save("../models/waste_classifier_mobilenet.h5")
print("✅ High-accuracy model saved to ../models/waste_classifier_mobilenet.h5")