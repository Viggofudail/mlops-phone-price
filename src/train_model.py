import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_PATH = os.path.join("models", "price_range_model.pkl")

def train():
    df = pd.read_csv(DATA_PATH)

    feature_cols = ["battery_power", "px_height", "px_width", "ram"]
    X = df[feature_cols]
    y = df["price_range"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    print("=== Classification Report ===")
    print(classification_report(y_val, y_pred))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
