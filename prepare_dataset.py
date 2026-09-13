from pathlib import Path
import random

from datasets import load_dataset
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "food101/data"
OUTPUT_DIR = Path("dataset")

RANDOM_SEED = 42

CLASSES = [
    "chicken_curry",
    "samosa",
    "fried_rice",
    "spring_rolls",
    "dumplings",
    "omelette",
    "grilled_cheese_sandwich",
    "chicken_wings",
    "hamburger",
    "pizza",
    "french_fries",
    "pancakes",
    "ice_cream",
    "cheesecake",
    "apple_pie",
]

TRAIN_PER_CLASS = 600
VALIDATION_PER_CLASS = 150
TEST_PER_CLASS = 250


# --------------------------------------------------
# Create output directories
# --------------------------------------------------

for split in ["train", "validation", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# --------------------------------------------------
# Load Food-101 Parquet dataset
# --------------------------------------------------

print("Loading Food-101...")

dataset = load_dataset(
    "parquet",
    data_files={
        "train": f"{DATA_PATH}/train-*.parquet",
        "validation": f"{DATA_PATH}/validation-*.parquet",
    }
)

print("Dataset loaded.")


# --------------------------------------------------
# Convert class names to label IDs
# --------------------------------------------------

label_names = dataset["train"].features["label"].names

class_to_id = {
    class_name: label_names.index(class_name)
    for class_name in CLASSES
}

print("\nSelected classes:")

for class_name, label_id in class_to_id.items():
    print(f"{class_name:30} -> {label_id}")


# --------------------------------------------------
# Collect image indices by class
# --------------------------------------------------

train_indices = {
    class_name: []
    for class_name in CLASSES
}

test_indices = {
    class_name: []
    for class_name in CLASSES
}


print("\nFinding training images...")

for index, label in enumerate(dataset["train"]["label"]):

    class_name = label_names[label]

    if class_name in train_indices:
        train_indices[class_name].append(index)


print("Finding test images...")

for index, label in enumerate(dataset["validation"]["label"]):

    class_name = label_names[label]

    if class_name in test_indices:
        test_indices[class_name].append(index)


# --------------------------------------------------
# Verify class counts
# --------------------------------------------------

print("\nClass counts:")

for class_name in CLASSES:

    train_count = len(train_indices[class_name])
    test_count = len(test_indices[class_name])

    print(
        f"{class_name:30} "
        f"train={train_count} "
        f"test={test_count}"
    )


# --------------------------------------------------
# Save training + validation images
# --------------------------------------------------

random.seed(RANDOM_SEED)

print("\nExporting training and validation images...")

for class_name in CLASSES:

    indices = train_indices[class_name]

    random.shuffle(indices)

    train_part = indices[:TRAIN_PER_CLASS]
    validation_part = indices[
        TRAIN_PER_CLASS:
        TRAIN_PER_CLASS + VALIDATION_PER_CLASS
    ]

    # Training images
    for counter, index in enumerate(train_part):

        image = dataset["train"][index]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        output_path = (
            OUTPUT_DIR
            / "train"
            / class_name
            / f"{counter:04d}.jpg"
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95
        )

    # Validation images
    for counter, index in enumerate(validation_part):

        image = dataset["train"][index]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        output_path = (
            OUTPUT_DIR
            / "validation"
            / class_name
            / f"{counter:04d}.jpg"
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95
        )


# --------------------------------------------------
# Save test images
# --------------------------------------------------

print("Exporting test images...")

for class_name in CLASSES:

    indices = test_indices[class_name]

    random.shuffle(indices)

    indices = indices[:TEST_PER_CLASS]

    for counter, index in enumerate(indices):

        image = dataset["validation"][index]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        output_path = (
            OUTPUT_DIR
            / "test"
            / class_name
            / f"{counter:04d}.jpg"
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95
        )


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n" + "=" * 50)
print("Dataset preparation completed!")
print("=" * 50)

print(f"Training images:    {len(CLASSES) * TRAIN_PER_CLASS}")
print(f"Validation images: {len(CLASSES) * VALIDATION_PER_CLASS}")
print(f"Test images:       {len(CLASSES) * TEST_PER_CLASS}")
print(f"Total images:      {len(CLASSES) * (TRAIN_PER_CLASS + VALIDATION_PER_CLASS + TEST_PER_CLASS)}")