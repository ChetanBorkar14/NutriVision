import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("dataset")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# --------------------------------------------------
# Load training dataset
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# Save class names BEFORE applying map/prefetch
class_names = train_ds.class_names


# --------------------------------------------------
# Load validation dataset
# --------------------------------------------------

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names
)


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names
)


# --------------------------------------------------
# Display class names
# --------------------------------------------------

print("\nClasses:")
print(class_names)

print("\nNumber of classes:")
print(len(class_names))


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


# --------------------------------------------------
# Prepare training dataset
# --------------------------------------------------

def preprocess_train(image, label):

    image = data_augmentation(image)
    image = preprocess_input(image)

    return image, label


# --------------------------------------------------
# Prepare validation/test datasets
# --------------------------------------------------

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

test_ds = test_ds.map(
    preprocess_eval,
    num_parallel_calls=tf.data.AUTOTUNE
)


# --------------------------------------------------
# Performance optimization
# --------------------------------------------------

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# Display dataset information
# --------------------------------------------------

print("\nDataset preparation completed!")

print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nTrain batches:")
print(tf.data.experimental.cardinality(train_ds).numpy())

print("\nValidation batches:")
print(tf.data.experimental.cardinality(validation_ds).numpy())

print("\nTest batches:")
print(tf.data.experimental.cardinality(test_ds).numpy())


# --------------------------------------------------
# Batch inspection
# --------------------------------------------------

images, labels = next(iter(train_ds))

print("\nBatch inspection")
print("=" * 50)

print("Image tensor shape:", images.shape)
print("Label tensor shape:", labels.shape)

print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)

print("Minimum pixel value:", tf.reduce_min(images).numpy())
print("Maximum pixel value:", tf.reduce_max(images).numpy())

print("\nFirst 10 labels:")
print(labels[:10].numpy())


# --------------------------------------------------
# Visualize augmented training images
# --------------------------------------------------

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        # Convert [-1, 1] back to [0, 1] for display
        display_image = (images[i] + 1.0) / 2.0

        plt.imshow(display_image)

        plt.title(
            class_names[labels[i].numpy()]
        )

        plt.axis("off")


plt.tight_layout()
plt.show()