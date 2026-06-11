import numpy as np
import tensorflow as tf
from PIL import Image

# load model
model = tf.keras.models.load_model("models/culture_image_model.h5")

class_names = [
    "CHITHIRAI THIRUVIZHA",
    "DIWALI",
    "HOLI",
    "JALLIKATU",
    "KARTHIGAI DEEPAM",
    "PONGAL",
    "RAKSHA BANDHAN",
    "THAIPUSAM",
    "THIRUVAIYARU"
]

def predict_image(img_path):

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))

    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    label = class_names[class_index]

    print("Prediction:", label, confidence)

    return label, confidence