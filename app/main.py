from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import os
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.path.join("models", "price_range_model.pkl")
DATA_PATH = os.path.join("data", "raw", "train.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
FEATURES = ["battery_power", "px_height", "px_width", "ram"]
feature_min = df[FEATURES].min()

def validate_min(value: int, feature_name: str):
    min_val = feature_min[feature_name]
    if value < min_val:
        raise ValueError(f"{feature_name} minimal adalah {min_val}, input anda: {value}")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None, "error": None})

@app.post("/", response_class=HTMLResponse)
async def post_predict(
    request: Request,
    ram: int = Form(...),
    battery_power: int = Form(...),
    px_width: int = Form(...),
    px_height: int = Form(...),
):
    try:
        validate_min(battery_power, "battery_power")
        validate_min(px_height, "px_height")
        validate_min(px_width, "px_width")
        validate_min(ram, "ram")

        features = np.array([[battery_power, px_height, px_width, ram]])
        prediction = model.predict(features)[0]

        PRICE_MAP = {
            0: "Rp.500.000 - 1.000.000",
            1: "Rp.1.000.000 - 2.000.000",
            2: "Rp.2.000.000 - 4.000.000",
            3: "Rp.4.000.000 ++"
        }

        predicted_price = PRICE_MAP.get(prediction, "Harga tidak diketahui")

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": predicted_price, "error": None},
        )
    except ValueError as ve:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": None, "error": str(ve)},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": None, "error": f"Error: {e}"},
        )
