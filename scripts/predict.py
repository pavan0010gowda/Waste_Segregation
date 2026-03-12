from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("../models/waste_classifier.h5")

img_path = "data/test/dry_decomposable/ther.jpg"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
labels = ['dry_decomposable', 'dry_recyclable', 'dry_non_degradable', 'wet']
print("Prediction:", labels[np.argmax(pred)])
