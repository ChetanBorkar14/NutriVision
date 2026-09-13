from datasets import load_dataset

DATA_PATH = "food101/data"

dataset = load_dataset(
    "parquet",
    data_files={
        "train": f"{DATA_PATH}/train-*.parquet",
        "validation": f"{DATA_PATH}/validation-*.parquet",
    }
)

print(dataset)

print("\nTrain features:")
print(dataset["train"].features)

print("\nValidation features:")
print(dataset["validation"].features)

print("\nFirst training example:")
print(dataset["train"][0])