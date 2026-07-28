from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = BASE_DIR / "random_forest_model.pkl"
DATA_FILENAME = BASE_DIR / "student_lifestyle_dataset.csv"


def load_dataset(path: str = DATA_FILENAME) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "Student_ID",
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA",
        "Stress_Level",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset tidak valid. Kolom yang hilang: {', '.join(sorted(missing_columns))}"
        )
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_save_model():
    print("1. Membaca dataset...")
    df = load_dataset()

    X = df.drop(columns=["Student_ID", "Stress_Level"])
    y = df["Stress_Level"]

    print("2. Membagi data training dan testing (80:20) dengan stratifikasi...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("3. Membangun pipeline dan melatih model Random Forest...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    print("4. Mengevaluasi performa model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- HASIL EVALUASI MODEL ---")
    print(f"Akurasi Model: {accuracy * 100:.2f}%\n")
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_FILENAME)
    print(f"5. Model berhasil disimpan sebagai '{MODEL_FILENAME}'")


if __name__ == "__main__":
    train_and_save_model()
