APLIKASI DATA SAINS - CARA MENJALANKAN (SINGKAT)
Judul: Analisis dan Prediksi Tingkat Kemiskinan Provinsi
Berdasarkan Ketersediaan Sarana Kesehatan Desa/Kelurahan

ISI FOLDER INI (4 file, jangan dipisah):
- app.py (aplikasi Streamlit)
- Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2025.xlsx
- Jumlah_Desa_Kelurahan_Yang_Memiliki_Sarana_Kesehatan_Menurut_Provinsi__2024.xlsx
- README_CARA_JALANIN.txt (file ini)

LANGKAH 1 - Buka terminal di folder ini
  cd <folder hasil unzip>

LANGKAH 2 - Buat virtual environment (sekali saja, supaya tidak bentrok
dengan Python sistem di Linux Mint/Ubuntu)
  python3 -m venv venv

LANGKAH 3 - Aktifkan virtual environment
  Linux/Mac:  source venv/bin/activate
  Windows:    venv\Scripts\activate

LANGKAH 4 - Install library (sekali saja)
  pip install streamlit pandas scikit-learn openpyxl matplotlib

LANGKAH 5 - Jalankan aplikasi
  streamlit run app.py

Browser akan membuka http://localhost:8501 secara otomatis.
Kalau tidak terbuka sendiri, salin alamat itu ke browser.

CARA DEMO SAAT PRESENTASI:
- Menu di kiri: Dashboard, Data & Normalisasi, Pemodelan, Prediksi.
- Dashboard: lihat metrik 38 provinsi dan Top 10.
- Pemodelan: klik tombol "Hitung evaluasi LOOCV", tunggu hasilnya.
  Random Forest adalah yang terbaik (MAE sekitar 3,87).
- Prediksi: isi 4 angka, klik tombol "Prediksi".
  Contoh: RS 119, Puskesmas 490, Pustu 995, Apotek 1248
  -> Random Forest 9,12%, Ridge 9,61%.
  Contoh provinsi tipikal: RS 42, Puskesmas 208, Pustu 442, Apotek 262.

CATATAN PENTING:
- Prediksi ini indikatif saja, bukan dasar anggaran.
  MAE sekitar 3,87 poin, jadi nilai sebenarnya bisa meleset beberapa poin.
- Untuk menghentikan aplikasi: tekan Ctrl+C di terminal.
- Untuk menjalankan lagi besok: ulangi Langkah 1, 3, dan 5 saja
  (venv dan library sudah terpasang, tidak perlu install ulang).
