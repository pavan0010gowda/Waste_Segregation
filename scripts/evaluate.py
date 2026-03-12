# scripts/evaluate.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

valid_dir = "data/test"
valid_gen = ImageDataGenerator(rescale=1./255)
valid_data = valid_gen.flow_from_directory(
    valid_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

model = load_model("../models/waste_classifier.h5")
loss, accuracy = model.evaluate(valid_data)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
