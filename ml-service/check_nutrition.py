import pandas as pd
from pathlib import Path


NUTRITION_PATH = Path("../nutrition/nutrition.csv")
PORTION_PATH = Path("../nutrition/portion.csv")


nutrition = pd.read_csv(NUTRITION_PATH)
portion = pd.read_csv(PORTION_PATH)


print("\nNutrition Database")
print("=" * 60)
print("Number of foods:", len(nutrition))
print("\nColumns:")
print(nutrition.columns.tolist())

print("\nFoods:")
for food in nutrition["food"]:
    print("-", food)


print("\nPortion Database")
print("=" * 60)
print("Number of foods:", len(portion))
print("\nColumns:")
print(portion.columns.tolist())


# Check food names
nutrition_foods = set(nutrition["food"])
portion_foods = set(portion["food"])

if nutrition_foods == portion_foods:
    print("\nFood matching check: PASSED")
else:
    print("\nFood matching check: FAILED")

    print("\nOnly in nutrition.csv:")
    print(nutrition_foods - portion_foods)

    print("\nOnly in portion.csv:")
    print(portion_foods - nutrition_foods)


print("\nNutrition database verification completed!")