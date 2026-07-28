# uas-ml-stress-prediction

## Deskripsi

Aplikasi ini adalah prototipe Machine Learning untuk memprediksi tingkat stres mahasiswa berdasarkan data gaya hidup dan akademik. Model menggunakan Random Forest, sedangkan antarmuka pengguna dibangun dengan Streamlit.

## Cara mudah menjalankan program ini

### 1. Siapkan lingkungan kerja di VS Code

- Buka folder proyek di VS Code:
  `C:\Users\kalim\Downloads\uas-ML-main\uas-ML-main`
- Pastikan folder sudah berisi: `app.py`, `train.py`, `student_lifestyle_dataset.csv`, `requirements.txt`, `Dockerfile`, `startup.sh`.

### 2. Buat virtual environment dan install paket

Buka terminal PowerShell di VS Code dan jalankan:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Karena aplikasi ini dibuat dengan Python dan Streamlit, XAMPP tidak diperlukan.

### 3. Latih model terlebih dahulu

```powershell
python train.py
```

Penjelasan sederhana:
- `train.py` membaca file CSV dan melatih model Random Forest.
- Setelah selesai, model disimpan ke file `random_forest_model.pkl`.
- File model ini seperti resep yang dipakai `app.py` saat memprediksi.

### 4. Jalankan aplikasi

```powershell
streamlit run app.py
```

Buka browser dan akses:

```text
http://localhost:8501
```

Penjelasan singkat alur aplikasi:
- `app.py` menjalankan server Streamlit.
- Browser membuka halaman web lokal.
- Anda memasukkan data kebiasaan harian lewat sidebar.
- Klik tombol `Prediksi` untuk melihat hasil stres dan probabilitasnya.

## Struktur Proyek

- `app.py` - aplikasi Streamlit untuk user interface dan prediksi.
- `train.py` - script pelatihan model ML.
- `student_lifestyle_dataset.csv` - data sumber fitur.
- `requirements.txt` - paket Python yang dibutuhkan.
- `Dockerfile` - container image untuk deployment Azure Container App.
- `startup.sh` - perintah startup untuk Azure App Service.
- `.github/workflows/azure-container-deploy.yml` - alur GitHub Actions deploy otomatis.
- `.gitignore` - file yang mengecualikan file lokal yang tidak perlu didorong ke Git.

## Perbaikan UI Streamlit

Aplikasi sekarang memiliki:
- Navigasi `Prediksi`, `Data & EDA`, dan `Tentang`.
- Ringkasan input berupa metrik interaktif.
- Preview dataset dan distribusi kategori stres.
- Penjelasan hasil prediksi yang lebih jelas.

## Backend dan training

- `train.py` menggunakan pipeline Scikit-Learn:
  - `StandardScaler` untuk normalisasi fitur.
  - `RandomForestClassifier` untuk prediksi.
- `train.py` juga menggunakan `stratify=y` untuk menjaga proporsi label.
- Hasil pelatihan disimpan dalam `random_forest_model.pkl`.

## Menjalankan di Docker lokal

Jika ingin mencoba container lokal:

```powershell
docker build -t uas-ml-app .
docker run -p 8501:8501 uas-ml-app
```

Lalu buka:

```text
http://localhost:8501
```

## Deploy otomatis ke Azure Container Apps

Sebelum deploy, siapkan:
- Azure Container Registry (ACR)
- Azure Container Apps Environment
- Azure Container App
- GitHub secret:
  - `AZURE_CREDENTIALS`
  - `AZURE_RESOURCE_GROUP`
  - `AZURE_ACR_NAME`
  - `AZURE_CONTAINERAPP_NAME`

### Contoh pembuatan resource Azure

```powershell
az login
az group create --name uas-ml-rg --location southeastasia
az acr create --resource-group uas-ml-rg --name <ACR_NAME> --sku Standard
az containerapp env create --name uas-ml-env --resource-group uas-ml-rg --location southeastasia
az containerapp create --name <APP_NAME> --resource-group uas-ml-rg --environment uas-ml-env --image <ACR_NAME>.azurecr.io/uas-ml-app:latest --target-port 8501 --ingress external
```

### GitHub Actions deploy otomatis

Setelah membuat secret di GitHub, setiap dorongan ke cabang `main` akan:
1. checkout kode
2. install dependensi
3. melatih model
4. membangun Docker image
5. mengirim image ke ACR
6. memperbarui Azure Container App dengan image terbaru

## Sinkronisasi VS Code, GitHub, dan Azure

1. Kerja di VS Code: edit file, simpan, dan jalankan lokal.
2. Commit perubahan ke Git:
   ```powershell
git add .
git commit -m "Setup Azure container deploy"
```
3. Push ke GitHub:
   ```powershell
git push -u origin main
```
4. GitHub Actions otomatis akan berjalan ketika ada push ke `main`.
5. Azure menerima update image secara otomatis.

## Catatan penting

- `random_forest_model.pkl` tidak perlu di-commit ke Git.
- Jika ingin deployment App Service saja, gunakan `startup.sh` dan Azure App Service Python.
- Jika ingin deployment Container Apps, gunakan `Dockerfile` dan workflow GitHub Actions.

## Troubleshooting sederhana

- Jika `streamlit` tidak ditemukan: pastikan virtual environment aktif.
- Jika `random_forest_model.pkl` tidak ada: jalankan `python train.py` dulu.
- Jika browser tidak muncul: buka manual `http://localhost:8501`.
