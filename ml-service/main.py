from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import pandas as pd
import numpy as np
import json
from pathlib import Path
import io


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "nutrivision_finetuned.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
NUTRITION_PATH = BASE_DIR / "nutrition" / "nutrition.csv"
PORTION_PATH = BASE_DIR / "nutrition" / "portion.csv"


# --------------------------------------------------
# Load model and data once when server starts
# --------------------------------------------------

print("Loading NutriVision model...")

model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

nutrition_df = pd.read_csv(NUTRITION_PATH)
portion_df = pd.read_csv(PORTION_PATH)

print("Model loaded successfully.")
print("Number of classes:", len(class_names))


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(title="NutriVision ML Service")


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Root / Health
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "NutriVision",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": len(class_names)
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    portion: str = "medium"
):

    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # Check portion
    valid_portions = {
        "small": "small_g",
        "medium": "medium_g",
        "large": "large_g",
        "one_person": "one_person_g",
        "two_person": "two_person_g"
    }

    if portion not in valid_portions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid portion. Choose from: {list(valid_portions.keys())}"
        )

    # Read image
    contents = await file.read()

    try:
        img = image.load_img(
            io.BytesIO(contents),
            target_size=(224, 224)
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to process the uploaded image."
        )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing
    img_array = preprocess_input(img_array)

    # Model prediction
    predictions = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    predicted_food = class_names[predicted_index]

    low_confidence = confidence < 0.60

    # --------------------------------------------------
    # Find nutrition information
    # --------------------------------------------------

    nutrition_row = nutrition_df[
        nutrition_df["food"] == predicted_food
    ]

    portion_row = portion_df[
        portion_df["food"] == predicted_food
    ]

    if nutrition_row.empty or portion_row.empty:
        raise HTTPException(
            status_code=500,
            detail=f"Nutrition data not found for {predicted_food}"
        )

    nutrition_row = nutrition_row.iloc[0]
    portion_row = portion_row.iloc[0]

    # Selected portion weight
    portion_column = valid_portions[portion]
    weight_g = float(portion_row[portion_column])

    # --------------------------------------------------
    # Calculate nutrition
    # --------------------------------------------------

    multiplier = weight_g / 100

    calories = float(nutrition_row["calories"]) * multiplier
    protein = float(nutrition_row["protein_g"]) * multiplier
    carbs = float(nutrition_row["carbs_g"]) * multiplier
    fat = float(nutrition_row["fat_g"]) * multiplier
    fiber = float(nutrition_row["fiber_g"]) * multiplier
    sugars = float(nutrition_row["sugars_g"]) * multiplier
    saturated_fat = float(nutrition_row["saturated_fat_g"]) * multiplier
    sodium = float(nutrition_row["sodium_mg"]) * multiplier

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {
    "food": predicted_food,
    "confidence": round(confidence, 4),
    "low_confidence": low_confidence,
    "portion": portion,
    "weight_g": weight_g,
    "insights": {
        "food_note": nutrition_row["food_note"],
        "balance_tip": nutrition_row["balance_tip"]
    },
    "nutrition": {
        "calories": round(calories, 2),
        "protein_g": round(protein, 2),
        "carbs_g": round(carbs, 2),
        "fat_g": round(fat, 2),
        "fiber_g": round(fiber, 2),
        "sugars_g": round(sugars, 2),
        "saturated_fat_g": round(saturated_fat, 2),
        "sodium_mg": round(sodium, 2)
    }
}
