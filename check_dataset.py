from pathlib import Path

DATASET_DIR = Path("dataset")

expected = {
    "train": 600,
    "validation": 150,
    "test": 250,
}

print("\nDataset verification")
print("=" * 60)

all_correct = True

for split, expected_count in expected.items():

    print(f"\n{split.upper()}")

    split_total = 0

    for class_dir in sorted((DATASET_DIR / split).iterdir()):

        if not class_dir.is_dir():
            continue

        count = len(list(class_dir.glob("*.jpg")))

        print(f"{class_dir.name:30} {count}")

        split_total += count

        if count != expected_count:
            all_correct = False

    print(f"Total: {split_total}")


print("\n" + "=" * 60)

if all_correct:
    print("DATASET VERIFICATION PASSED")
else:
    print("DATASET VERIFICATION FAILED")