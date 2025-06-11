from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import numpy as np
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Paths
MODEL_PATH = os.path.join("models", "price_range_model.pkl")
ACCURACY_PATH = os.path.join("models", "accuracy.txt")
META_PATH = os.path.join("models", "meta.json")

# Load model & metadata
model = joblib.load(MODEL_PATH)

with open(META_PATH, "r") as f:
    meta = json.load(f)

chipset_list = meta.get("chipset_list", [])
resolution_list = meta.get("resolution_list", ["720p", "1080p", "2k+"])
label_mapping = meta.get("label_mapping", {})
label_map = {int(k): v for k, v in label_mapping.items()}  # pastikan int untuk key

# Fungsi skor chipset
def chipset_score(chipset: str) -> int:
    chipset = chipset.lower()
    if 'snapdragon 8 gen 3' in chipset:
        return 850
    elif 'snapdragon 8 gen 2' in chipset:
        return 820
    elif 'snapdragon 888' in chipset:
        return 800
    elif 'snapdragon 855' in chipset:
        return 730
    elif 'snapdragon 778' in chipset:
        return 720
    elif 'snapdragon 765' in chipset:
        return 690
    elif 'helio g99' in chipset:
        return 650
    elif 'tensor g4' in chipset:
        return 830
    elif 'tensor g3' in chipset:
        return 800
    elif 'tensor g2' in chipset:
        return 780
    elif 'tensor' in chipset:
        return 750
    elif 'apple a18' in chipset:
        return 870
    elif 'apple a17' in chipset:
        return 850
    elif 'apple a16' in chipset:
        return 830
    elif 'apple a15' in chipset:
        return 800
    elif 'apple a14' in chipset:
        return 770
    elif 'apple a13' in chipset:
        return 740
    elif 'apple a12' in chipset:
        return 720
    elif 'apple a11' in chipset:
        return 690
    elif 'kirin' in chipset:
        return 500
    elif 'exynos' in chipset:
        return 650
    else:
        return 400

# Konversi resolusi ke nilai numerik
def resolution_to_value(res_str: str) -> int:
    if res_str == "720p":
        return 720
    elif res_str == "1080p":
        return 1080
    elif res_str == "2k+":
        return 2000
    else:
        return 720

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    acc = None
    try:
        with open(ACCURACY_PATH, "r") as f:
            acc = round(float(f.read()) * 100, 2)
    except:
        acc = None
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "error": None,
        "accuracy": acc,
        "chipset_list": chipset_list,
        "resolution_list": resolution_list,
        "selected_chipset": None,
        "selected_resolution": None,
        "ram": None,
        "storage": None
    })

@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    ram: int = Form(...),
    storage: int = Form(...),
    display_resolution: str = Form(...),
    chipset: str = Form(...)
):
    try:
        display_res_value = resolution_to_value(display_resolution)
        chipset_val = chipset_score(chipset)

        # Fitur mentah sebelum ke model pipeline
        input_features_raw = np.array([[ram, storage, display_res_value, chipset_val]])
        print(f"DEBUG: Input Features (Raw for Pipeline): {input_features_raw}")

        prediction_numeric = model.predict(input_features_raw)[0]
        print(f"DEBUG: Raw Numeric Prediction from Model: {prediction_numeric}")
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_features_raw)[0]
            print(f"DEBUG: Prediction Probabilities: {probabilities}")

        prediction_label = label_map.get(int(prediction_numeric), "Unknown")
        print(f"DEBUG: Predicted Label: {prediction_label}")

        with open(ACCURACY_PATH, "r") as f:
            acc = round(float(f.read()) * 100, 2)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction_label,
            "error": None,
            "accuracy": acc,
            "chipset_list": chipset_list,
            "resolution_list": resolution_list,
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "ram": ram,
            "storage": storage
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "error": str(e),
            "accuracy": None,
            "chipset_list": chipset_list,
            "resolution_list": resolution_list,
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "ram": ram,
            "storage": storage
        })
