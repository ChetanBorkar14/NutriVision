import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("../dataset")
MODEL_PATH = Path("../models/nutrivision_finetuned.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# Load test dataset
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
for i, class_name in enumerate(class_names):
    print(f"{i:2} -> {class_name}")


# ============================================================
# Preprocessing
# ============================================================

preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


def preprocess_test(image, label):
    image = preprocess_input(image)
    return image, label


test_ds = test_ds.map(
    preprocess_test,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# ============================================================
# Load fine-tuned model
# ============================================================

print("\nLoading fine-tuned model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# Evaluate
# ============================================================

print("\n" + "=" * 70)
print("Final Test Evaluation")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")
print(f"{test_accuracy * 100:.2f}%")


# ============================================================
# Generate predictions
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Classification Report
# ============================================================

print("\n" + "=" * 70)
print("Classification Report")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 70)
print("Confusion Matrix")
print("=" * 70)

print(cm)


# ============================================================
# Plot Confusion Matrix
# ============================================================

plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.title("NutriVision Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.colorbar()

plt.tight_layout()

plt.show()