import pandas as pd
import os
import joblib
import json
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_PATH = os.path.join("models", "price_range_model.pkl")
ACCURACY_PATH = os.path.join("models", "accuracy.txt")
META_PATH = os.path.join("models", "meta.json")

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
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df['display_resolution_cat'] = df['display_resolution'].apply(resolution_category)
    df['chipset_score'] = df['chipset'].apply(chipset_score)

    features = ['ram', 'storage', 'display_resolution_cat', 'chipset_score']
    X = df[features].copy()
    y = df['price_range']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Ganti ini dengan URL MLflow server Railway kamu
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        print("=== Classification Report ===\n", report)
        print(f"Accuracy: {acc:.4f}")

        # Save model lokal
        joblib.dump(pipeline, MODEL_PATH)

        # Log model & metric ke MLflow server
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_metric("accuracy", acc)

        with open(ACCURACY_PATH, "w") as f:
            f.write(str(acc))

        chipset_list = sorted(df['chipset'].dropna().unique().tolist())
        resolution_list = ["720p", "1080p", "2k+"]

        meta = {
            "chipset_list": chipset_list,
            "resolution_list": resolution_list
        }

        with open(META_PATH, "w") as f:
            json.dump(meta, f)

        print("Model, akurasi, dan meta (dropdown) berhasil disimpan.")

if __name__ == "__main__":
    train()
