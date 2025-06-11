import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
# PERUBAHAN: Tambahkan f1_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
# PERUBAHAN: Tambahkan SMOTE
from imblearn.over_sampling import SMOTE

DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_data(df):
    # Konversi resolusi ke nilai numerik
    def resolution_to_value(res_str):
        if res_str == "720p":
            return 720
        elif res_str == "1080p":
            return 1080
        elif res_str == "2k+":
            return 2000
        else:
            return 720

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

    df['display_resolution_cat'] = df['display_resolution'].map(resolution_to_value)
    df['chipset_score'] = df['chipset'].map(chipset_score)

    return df[['ram', 'storage', 'display_resolution_cat', 'chipset_score']], df['price_range']

def train():
    df = pd.read_csv(DATA_PATH)

    # Konversi label string ke angka
    label_mapping = {label: idx for idx, label in enumerate(df['price_range'].unique())}
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    df['price_range'] = df['price_range'].map(label_mapping)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Distribusi kelas di y_train SEBELUM SMOTE (proporsi):")
    print(y_train.value_counts(normalize=True))
    
    if len(y_train.value_counts()) > 1 and not y_train.empty:
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print("\nDistribusi kelas di y_train SETELAH SMOTE (proporsi):")
            print(y_train_resampled.value_counts(normalize=True))
        except ValueError as e:
            print(f"\nError saat menerapkan SMOTE: {e}. Menggunakan data training asli.")
            X_train_resampled, y_train_resampled = X_train, y_train
    else:
        print("\nTidak menerapkan SMOTE (y_train kosong atau hanya 1 kelas). Menggunakan data training asli.")
        X_train_resampled, y_train_resampled = X_train, y_train


    print("-" * 30)

    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight='balanced', random_state=42))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42))
        ])
    }

    best_f1_score = 0
    best_model = None
    best_name = ""

    mlflow.set_experiment("PhonePricePrediction")

    for name, model_pipeline in models.items():
        with mlflow.start_run(run_name=name):
            model_pipeline.fit(X_train_resampled, y_train_resampled)
            preds = model_pipeline.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score_weighted", f1)

            print(f"Model: {name}, Accuracy: {acc:.4f}, F1-score (weighted): {f1:.4f}")

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model_pipeline
                best_name = name

    # Simpan model terbaik
    joblib.dump(best_model, os.path.join(MODEL_DIR, "price_range_model.pkl"))

    # Simpan skor terbaik (sekarang F1 score)
    with open(os.path.join(MODEL_DIR, "accuracy.txt"), "w") as f:
        f.write(str(best_f1_score))

    # Simpan metadata
    meta = {
        "chipset_list": sorted(df['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"],
        "best_model_name": best_name,
        "best_model_metric_score": best_f1_score,
        "metric_used": "f1_score_weighted", 
        "label_mapping": inverse_mapping
    }

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"Training selesai. Model terbaik: {best_name} (F1-score weighted: {best_f1_score:.4f})")

if __name__ == "__main__":
    train()