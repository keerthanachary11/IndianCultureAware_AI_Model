import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# 1️⃣ Dataset path
train_dir = "dataset/train"

# 2️⃣ Data generator (rescale images)
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical"
)

# 3️⃣ Load MobileNetV2 without top layers
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# 4️⃣ Freeze base model layers (feature extraction)
base_model.trainable = False

# 5️⃣ Add custom top layers
x = layers.GlobalAveragePooling2D()(base_model.output)
output = layers.Dense(train_data.num_classes, activation="softmax")(x)

# 6️⃣ Create final model
model = models.Model(inputs=base_model.input, outputs=output)

# 7️⃣ Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 8️⃣ Train model
model.fit(train_data, epochs=5)

# 9️⃣ Save trained model
model.save("culture_image_model.h5")
