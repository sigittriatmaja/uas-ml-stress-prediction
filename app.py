import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = BASE_DIR / "random_forest_model.pkl"
DATA_FILENAME = BASE_DIR / "student_lifestyle_dataset.csv"

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_dataset(path: str = DATA_FILENAME) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)
    expected_columns = {
        "Student_ID",
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA",
        "Stress_Level",
    }
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Dataset tidak valid. Kolom yang hilang: "
            + ", ".join(sorted(missing_columns))
        )

    return df


@st.cache_resource
def load_model(path: str = MODEL_FILENAME):
    if not os.path.exists(path):
        try:
            from train import train_and_save_model

            train_and_save_model()
        except Exception as exc:
            raise FileNotFoundError(
                "Model belum ditemukan dan pelatihan otomatis gagal. Jalankan `python train.py` terlebih dahulu."
            ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Model belum ditemukan. Jalankan `python train.py` terlebih dahulu."
        )
    return joblib.load(path)


def build_input_dataframe() -> pd.DataFrame:
    study_hours = st.sidebar.slider(
        "📚 Jam Belajar per Hari",
        min_value=0.0,
        max_value=12.0,
        value=6.0,
        step=0.1,
    )
    extracurricular_hours = st.sidebar.slider(
        "🎨 Jam Ekstrakurikuler per Hari",
        min_value=0.0,
        max_value=8.0,
        value=2.0,
        step=0.1,
    )
    sleep_hours = st.sidebar.slider(
        "😴 Jam Tidur per Hari",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.1,
    )
    social_hours = st.sidebar.slider(
        "👥 Jam Bersosialisasi per Hari",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
    )
    physical_hours = st.sidebar.slider(
        "🏃 Jam Olahraga per Hari",
        min_value=0.0,
        max_value=8.0,
        value=2.0,
        step=0.1,
    )
    gpa = st.sidebar.slider(
        "📊 Indeks Prestasi Kumulatif (IPK/GPA)",
        min_value=0.00,
        max_value=4.00,
        value=3.10,
        step=0.01,
    )

    data = {
        "Study_Hours_Per_Day": study_hours,
        "Extracurricular_Hours_Per_Day": extracurricular_hours,
        "Sleep_Hours_Per_Day": sleep_hours,
        "Social_Hours_Per_Day": social_hours,
        "Physical_Activity_Hours_Per_Day": physical_hours,
        "GPA": gpa,
    }
    return pd.DataFrame([data])


def show_overview(df: pd.DataFrame):
    st.title("📊 Data & Eksplorasi")
    st.write(
        "Halaman ini membantu Anda memahami dataset, distribusi fitur, dan sebaran label stres."
    )

    total = len(df)
    value_counts = df["Stress_Level"].value_counts()
    high_count = int(value_counts.get("High", 0))
    moderate_count = int(value_counts.get("Moderate", 0))
    low_count = int(value_counts.get("Low", 0))

    st.markdown("**Ringkasan Dataset**")
    st.metric("Total Sampel", total)
    col1, col2, col3 = st.columns(3)
    col1.metric("High", high_count, "Kategori stres tinggi")
    col2.metric("Moderate", moderate_count, "Kategori stres sedang")
    col3.metric("Low", low_count, "Kategori stres rendah")

    st.subheader("Distribusi Kategori Tingkat Stres")
    count_df = df["Stress_Level"].value_counts().rename_axis("Stress_Level").reset_index(name="Jumlah")
    st.bar_chart(data=count_df.set_index("Stress_Level"))

    st.subheader("Statistik Deskriptif Fitur Numerik")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Pola Input Mahasiswa")
    selected_columns = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "GPA",
    ]
    st.line_chart(df[selected_columns])


def show_prediction_page(model, df: pd.DataFrame):
    st.title("🔮 Prediksi Tingkat Stres Mahasiswa")
    st.write(
        "Gunakan pola kebiasaan harian dan IPK untuk memprediksi level stres Anda."
    )

    st.sidebar.subheader("Input Kebiasaan Harian")
    st.sidebar.write(
        "Atur nilai input kemudian tekan tombol Prediksi untuk melihat hasilnya."
    )

    input_data = build_input_dataframe()
    st.subheader("📋 Ringkasan Data Input")
    st.dataframe(input_data, use_container_width=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Jam Belajar", f"{input_data.at[0, 'Study_Hours_Per_Day']} jam")
    metric_col2.metric("Jam Tidur", f"{input_data.at[0, 'Sleep_Hours_Per_Day']} jam")
    metric_col3.metric("GPA", f"{input_data.at[0, 'GPA']:.2f}")

    st.info("Masukkan kebiasaan harian Anda di panel kiri, lalu tekan tombol Prediksi untuk melihat hasilnya.")

    if st.button("🔮 Prediksi Tingkat Stres", use_container_width=True):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")

        if prediction == "Low":
            st.success(
                "Tingkat Stres Terprediksi: **Low (Rendah)** 😊\n\n"
                "Pola hidup Anda relatif seimbang. Pertahankan kebiasaan ini."
            )
        elif prediction == "Moderate":
            st.warning(
                "Tingkat Stres Terprediksi: **Moderate (Sedang)** 😐\n\n"
                "Jaga keseimbangan antara belajar, istirahat, dan waktu sosial."
            )
        else:
            st.error(
                "Tingkat Stres Terprediksi: **High (Tinggi)** ⚠️\n\n"
                "Pertimbangkan menurunkan beban kegiatan dan fokus pada kualitas tidur."
            )

        st.write("---")
        st.write("**Probabilitas setiap kategori:**")
        prob_df = pd.DataFrame(
            {"Tingkat Stres": classes, "Probabilitas (%)": probabilities * 100}
        )
        st.dataframe(
            prob_df.style.format({"Probabilitas (%)": "{:.1f}%"}),
            use_container_width=True,
        )

        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            st.subheader("🔧 Pentingnya Fitur Model")
            classifier = model.named_steps["clf"]
            feature_names = input_data.columns.tolist()
            importance_df = pd.DataFrame(
                {
                    "Fitur": feature_names,
                    "Importance": classifier.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Fitur"))


def show_about():
    st.title("🧠 Tentang Aplikasi")
    st.write(
        "Aplikasi Streamlit ini menggunakan model Random Forest untuk memprediksi tingkat stres siswa "
        "berdasarkan jam belajar, jam ekstrakurikuler, jam tidur, aktivitas sosial, olahraga, dan GPA."
    )
    st.write(
        "Gunakan menu di samping untuk melihat data, melakukan prediksi, dan mempelajari arsitektur aplikasi."
    )

    st.markdown("---")
    st.write("**Backend / Machine Learning**")
    st.write(
        "- `train.py` melakukan pelatihan model, memvalidasi performa, dan menyimpan model sebagai `random_forest_model.pkl`."
    )
    st.write("- `app.py` memuat model yang sudah dilatih dan menjalankan antarmuka prediksi di Streamlit.")
    st.write("- Pipeline model mencakup normalisasi fitur dan Random Forest untuk mengurangi overfitting.")

    st.markdown("---")
    st.write("**Frontend**")
    st.write("- Tampilan dibuat dengan Streamlit, cocok untuk prototipe data science dan aplikasi ML interaktif.")
    st.write("- Halaman `Data & EDA` membantu melihat distribusi dan statistik dataset sebelum prediksi.")

    st.markdown("---")
    st.write("**Deployment Azure**")
    st.write(
        "Untuk produksi, deploy aplikasi ini menggunakan Azure App Service Linux atau Azure Container App."
    )


def main():
    try:
        df = load_dataset()
    except Exception as exc:
        st.error(f"❌ Gagal memuat dataset: {exc}")
        st.stop()

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.warning(str(exc))
        model = None

    st.sidebar.title("📌 Menu Aplikasi")
    st.sidebar.markdown(
        "Gunakan menu ini untuk beralih antara Prediksi, Data & EDA, dan Tentang."
    )

    page = st.sidebar.selectbox(
        "Navigasi",
        ["Prediksi", "Data & EDA", "Tentang"],
    )

    if page == "Prediksi":
        if model is None:
            st.error(
                "Model tidak tersedia. Jalankan `python train.py` terlebih dahulu dan muat ulang halaman."
            )
        else:
            show_prediction_page(model, df)
    elif page == "Data & EDA":
        show_overview(df)
    else:
        show_about()


if __name__ == "__main__":
    main()
