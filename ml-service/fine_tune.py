import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("../dataset")
MODEL_DIR = Path("../models")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 15

EPOCHS = 8
LEARNING_RATE = 1e-5

STAGE1_MODEL = MODEL_DIR / "nutrivision_head.keras"
FINETUNED_MODEL = MODEL_DIR / "nutrivision_finetuned.keras"


# ============================================================
# Load datasets
# ============================================================

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


# ============================================================
# Data augmentation and preprocessing
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def preprocess_train(image, label):
    image = data_augmentation(image)
    image = preprocess_input(image)
    return image, label


def preprocess_eval(image, label):
    image = preprocess_input(image)
    return image, label


train_ds = train_ds.map(
    preprocess_train,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_ds = validation_ds.map(
    preprocess_eval,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# Rebuild MobileNetV2 model
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# ============================================================
# Build complete model
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="food_image"
)

x = base_model(inputs, training=False)

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


# ============================================================
# Load Stage 1 trained weights
# ============================================================

print("\nLoading Stage 1 model weights...")

stage1_model = tf.keras.models.load_model(STAGE1_MODEL)

model.set_weights(stage1_model.get_weights())

print("Stage 1 weights loaded successfully!")


# ============================================================
# Freeze everything first
# ============================================================

base_model.trainable = False


# ============================================================
# Unfreeze top layers of MobileNetV2
# ============================================================

fine_tune_at = 100

for layer in base_model.layers[fine_tune_at:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True


# ============================================================
# Compile with low learning rate
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# Display fine-tuning information
# ============================================================

trainable_count = sum(
    1 for layer in model.layers if layer.trainable
)

trainable_params = sum(
    layer.count_params()
    for layer in model.layers
    if layer.trainable
)

total_params = model.count_params()

print("\n" + "=" * 70)
print("NutriVision Stage 2: Fine-Tuning")
print("=" * 70)

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(f"{i:2} -> {class_name}")

print("\nFine-tuning configuration:")
print("Fine-tune from MobileNetV2 layer:", fine_tune_at)
print("Learning rate:", LEARNING_RATE)
print("Epochs:", EPOCHS)

print("\nModel parameters:")
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
print("Trainable layers:", trainable_count)


# ============================================================
# Callbacks
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=FINETUNED_MODEL,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# Fine-tuning
# ============================================================

print("\nStarting Stage 2 fine-tuning...\n")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# Final result
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)

print("\n" + "=" * 70)
print("Stage 2 Fine-Tuning Completed")
print("=" * 70)

print("\nBest validation accuracy:")
print(f"{best_val_accuracy:.4f}")

print("\nBest validation loss:")
print(f"{best_val_loss:.4f}")

print("\nBest fine-tuned model saved at:")
print(FINETUNED_MODEL)