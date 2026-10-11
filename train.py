import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models

print("Loading Cat and Dog dataset...")

DATASET_DIR = Path("dataset")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

if not (DATASET_DIR / "Cats").exists():
    print("ERROR: Cat folder not found inside dataset!")
    raise SystemExit(1)

if not (DATASET_DIR / "Dogs").exists():
    print("ERROR: Dog folder not found inside dataset!")
    raise SystemExit(1)

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_DIR),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_DIR),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Classes:", train_ds.class_names)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

Path("model").mkdir(exist_ok=True)
model.save("model/cat_dog_model.keras")

print("Training completed!")
print("Model saved: model/cat_dog_model.keras")