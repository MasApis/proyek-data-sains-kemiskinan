# Analisis dan Prediksi Tingkat Kemiskinan Provinsi

Berdasarkan ketersediaan sarana kesehatan desa/kelurahan di Indonesia.

**Pertanyaan:** Apakah provinsi yang desanya banyak punya fasilitas kesehatan, kemiskinannya lebih rendah, dan bisakah kita menebak kemiskinan itu hanya dari data fasilitas kesehatan?

**Jawaban singkat:** Ada hubungan, tapi lemah dan negatif. Model terbaik (Random Forest) masih meleset rata-rata sekitar 3,87 poin persen, jadi model ini alat bantu lihat pola, bukan dasar anggaran sendirian.

## Data

Sumber: Badan Pusat Statistik (BPS), situs resmi pemerintah.

- `Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2025.xlsx` — kemiskinan September 2025
- `Jumlah_Desa_Kelurahan_Yang_Memiliki_Sarana_Kesehatan_Menurut_Provinsi__2024.xlsx` — sarana kesehatan 2024

Setelah cleaning: 38 provinsi (baris Indonesia dibuang karena itu total nasional), 0 nilai kosong.
Total validasi: RS 2.637, Puskesmas 10.280, Pustu 24.383, Apotek 17.636.

Fitur model (hanya 4 ini): `desa_rs`, `desa_puskesmas`, `desa_pustu`, `desa_apotek`.
Target: `persen_miskin`.

Garis kemiskinan dan jumlah penduduk miskin sengaja TIDAK dipakai di model utama, karena turunan dari target (risiko leakage).

## Hasil utama (LOOCV, 38 fold)

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Ridge (laporan) | 4,8248 | 6,2629 | -0,0251 |
| Random Forest (laporan) | 3,7926 | 4,9744 | 0,3533 |
| Random Forest (hitung ulang Colab/Streamlit) | 3,8682 | 5,0423 | 0,3356 |
| Baseline rata-rata | 4,7939 | 6,3530 | -0,0548 |

Selisih kecil Random Forest antara laporan dan hitung ulang karena versi scikit-learn berbeda. Urutan dan kesimpulan sama: Random Forest terbaik, Ridge kalah dari baseline.

Feature importance Random Forest: apotek 0,745, RS 0,145, Puskesmas 0,084, Pustu 0,027.
Hati-hati: apotek kemungkinan besar proksi ukuran wilayah (Jawa vs Papua), bukan bukti sebab-akibat.

Median kemiskinan 9,345%, mean 10,27%, terendah Bali 3,42%, tertinggi Papua Tengah 29,45%.

## Aplikasi Streamlit

Jalankan dari folder ini (lihat juga `README_CARA_JALANIN.txt`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install streamlit pandas scikit-learn openpyxl matplotlib
streamlit run app.py
```

Browser akan membuka `http://localhost:8501`.

Halaman aplikasi:
- **Dashboard** — ringkasan 38 provinsi, Top 10, histogram
- **Data & Normalisasi** — tabel gabungan, statistik, contoh z-score, validasi total
- **Pemodelan** — klik tombol Hitung evaluasi LOOCV, lihat MAE/RMSE/R2, 8 kesalahan terbesar, feature importance
- **Prediksi** — isi 4 angka sarana, klik Prediksi

Contoh demo yang sudah dicoba:
- Median: RS 42, Puskesmas 208, Pustu 442, Apotek 262 → sekitar 9–10%
- RS 119, Puskesmas 490, Pustu 995, Apotek 1248 → Random Forest 9,12%, Ridge 9,61%
- Papua Tengah: RS 14, Puskesmas 98, Pustu 148, Apotek 45 → aslinya 29,45%, model sekitar 17% (contoh keterbatasan: model meremehkan ekstrem)

Prediksi ini indikatif saja, bukan dasar anggaran.

## E-book kelompok

Baca `ebook/EBook_Kelompok_Data_Sains_Kemiskinan.pdf` (30 halaman) untuk penjelasan awam lengkap:
glosarium, data, cleaning, EDA dengan 5 grafik, model, aplikasi, kode, keterbatasan,
35 pertanyaan dosen beserta jawabannya, dan skrip presentasi 6 orang fokus aplikasi.

Versi DOCX-nya juga ada di folder `ebook/` kalau kelompok mau mengedit.

## Notebook Colab

`notebook/Proyek_Data_Sains_Kemiskinan.ipynb` berisi langkah yang sama dengan aplikasi,
dalam bentuk notebook: load + cleaning, EDA, pemodelan LOOCV, dan prediksi interaktif.
Upload 2 file Excel ke Colab, lalu jalankan cell dari atas ke bawah.

## Struktur repo

```
app.py                                   # Aplikasi Streamlit
README_CARA_JALANIN.txt                  # Cara jalanin singkat
Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2025.xlsx
Jumlah_Desa_Kelurahan_Yang_Memiliki_Sarana_Kesehatan_Menurut_Provinsi__2024.xlsx
ebook/EBook_Kelompok_Data_Sains_Kemiskinan.pdf
ebook/EBook_Kelompok_Data_Sains_Kemiskinan.docx
notebook/Proyek_Data_Sains_Kemiskinan.ipynb
```

## Keterbatasan (jujur, untuk presentasi)

- Hanya pakai fasilitas kesehatan, tanpa pendidikan, ekonomi, dan geografi.
- Variabelnya jumlah desa, bukan rasio per penduduk.
- Data hanya 38 provinsi, tahun beda (2025 vs 2024), hasil asosiasi bukan sebab-akibat.
- Model masih meleset sekitar 4 poin, dan meremehkan provinsi ekstrem.

Rekomendasi: buat rasio cakupan, tambah indikator pendidikan dan ekonomi, dan pakai model ini hanya sebagai alat bantu prioritisasi investigasi.

## Laporan resmi

Laporan resmi kelompok (HDP_LABS_PROYEK_2.pdf) sudah dikumpulkan terpisah dan tidak diubah oleh repo ini.
Repo ini adalah paket kerja + aplikasi + e-book belajar kelompok, bukan pengganti laporan resmi.
