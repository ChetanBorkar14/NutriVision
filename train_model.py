import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

NUM_CLASSES = 15
EPOCHS = 10


# --------------------------------------------------
# Create model directory
# --------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

class_names = train_ds.class_names

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names
)


# --------------------------------------------------
# Data augmentation
# --------------------------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


# --------------------------------------------------
# MobileNetV2 preprocessing
# --------------------------------------------------

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def preprocess_train(image, label):

    image = data_augmentation(image)
    image = preprocess_input(image)

    return image, label


def preprocess_eval(image, label):

    image = preprocess_input(image)

    return image, label


# --------------------------------------------------
# Apply preprocessing
# --------------------------------------------------

train_ds = train_ds.map(
    preprocess_train,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_ds = validation_ds.map(
    preprocess_eval,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# Prefetch
# --------------------------------------------------

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# Load MobileNetV2
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# --------------------------------------------------
# Freeze base model
# --------------------------------------------------

base_model.trainable = False


# --------------------------------------------------
# Build model
# --------------------------------------------------

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="food_image"
)

x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="feature_dense"
)(x)

x = tf.keras.layers.Dropout(
    0.3,
    name="dropout"
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="food_prediction"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="NutriVision_MobileNetV2"
)


# --------------------------------------------------
# Compile
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

checkpoint_path = MODEL_DIR / "nutrivision_head.keras"

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\n" + "=" * 70)
print("Starting Stage 1 Training")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# --------------------------------------------------
# Training complete
# --------------------------------------------------

print("\n" + "=" * 70)
print("Stage 1 Training Completed")
print("=" * 70)

print("\nBest model saved at:")
print(checkpoint_path)

print("\nBest validation accuracy:")

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print(f"{best_val_accuracy:.4f}")