import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "price_range_model.pkl")
ACCURACY_PATH = os.path.join(MODEL_DIR, "accuracy.txt")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


def chipset_score(chipset: str) -> int:
    chipset = chipset.lower()
    mapping = {
        "snapdragon 8 gen 3": 850, "snapdragon 8 gen 2": 820, "snapdragon 888": 800,
        "snapdragon 855": 730, "snapdragon 778": 720, "snapdragon 765": 690,
        "helio g99": 650, "tensor g4": 830, "tensor g3": 800, "tensor g2": 780,
        "tensor": 750, "apple a18": 870, "apple a17": 850, "apple a16": 830,
        "apple a15": 800, "apple a14": 770, "apple a13": 740, "apple a12": 720,
        "apple a11": 690, "kirin": 500, "exynos": 650
    }
    for key in mapping:
        if key in chipset:
            return mapping[key]
    return 400


def resolution_category(res_str):
    try:
        parts = res_str.lower().replace(" ", "").split('x')
        width = int(parts[0])
        if width <= 720:
            return 720
        elif width <= 1080:
            return 1080
        else:
            return 2000
    except:
        return 720


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Feature engineering
    df['display_resolution_cat'] = df['display_resolution'].apply(resolution_category)
    df['chipset_score'] = df['chipset'].apply(chipset_score)

    features = ['ram', 'storage', 'display_resolution_cat', 'chipset_score']
    X = df[features].astype(float)  # Pastikan float untuk hindari warning MLflow
    y = df['price_range']

    # Split data stratify supaya distribusi kelas tetap terjaga
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Predict dan evaluasi
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)

    print("=== Classification Report ===")
    print(classification_report(y_val, y_pred))
    print(f"Accuracy: {acc:.4f}")

    # MLflow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment_name = "phone_price_classification"

    # Buat atau dapatkan experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Log ke MLflow
    with mlflow.start_run():
        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 100,
            "features": ", ".join(features)
        })
        mlflow.log_metric("accuracy", acc)
        # Log metrics detail class per class
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{cls}_{metric_name}", metric_value)

        # Signature dan example input (poin penting buat MLOps)
        input_example = X_train.head(3)
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

    # Save lokal
    joblib.dump(pipeline, MODEL_PATH)
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(acc))

    meta = {
        "chipset_list": sorted(df['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"]
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    print("Model, akurasi, dan metadata berhasil disimpan.")


if __name__ == "__main__":
    train()
