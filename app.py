"""
Aplikasi Data Sains - Analisis dan Prediksi Tingkat Kemiskinan Provinsi
Berdasarkan Ketersediaan Sarana Kesehatan Desa/Kelurahan di Indonesia.

Cara pakai (di folder yang sama dengan 2 file Excel BPS):
    pip install streamlit pandas scikit-learn openpyxl matplotlib
    streamlit run app.py

Alur aplikasi ini, sesuai laporan Bab 1-7:
    Dashboard            -> ringkasan + grafik utama
    Data & Normalisasi   -> data gabungan + contoh z-score
    Pemodelan            -> Ridge + Random Forest, evaluasi LOOCV
    Prediksi             -> isi 4 angka sarana, keluar prediksi persen miskin

Catatan penting:
    - Fitur model HANYA 4 sarana kesehatan. Garis kemiskinan dan jumlah
      penduduk miskin sengaja TIDAK dipakai di model, karena keduanya
      turunan dari targetnya sendiri (risiko leakage / mencontek jawaban).
    - Prediksi ini indikatif saja, bukan dasar anggaran. MAE Random Forest
      sekitar 3,87 poin, jadi nilai sebenarnya bisa meleset beberapa poin.
    - Data kemiskinan September 2025 digabung dengan sarana kesehatan 2024,
      jadi hasilnya asosiasi, bukan sebab-akibat.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Nama file Excel yang harus ada di folder yang sama dengan app.py.
# Nama ini persis seperti file BPS yang dipakai di laporan.
FILE_MISKIN = "Jumlah_dan_Persentase_Penduduk_Miskin_Menurut_Provinsi__2025.xlsx"
FILE_FASKES = "Jumlah_Desa_Kelurahan_Yang_Memiliki_Sarana_Kesehatan_Menurut_Provinsi__2024.xlsx"

FEATS = ["desa_rs", "desa_puskesmas", "desa_pustu", "desa_apotek"]
TARGET = "persen_miskin"


def load_data():
    """Baca, bersihkan, dan gabungkan 2 file Excel BPS (Bab 3 laporan).

    Langkahnya sama persis dengan yang sudah diverifikasi di Colab:
    baca sebagai teks, ambil 39 baris pertama (38 provinsi + Indonesia),
    buang baris Indonesia, pilih September 2025 dan 4 sarana, ubah nama
    kolom jadi snake_case, ubah teks jadi angka, lalu inner join.
    """
    miskin = pd.read_excel(FILE_MISKIN, dtype=str).iloc[:39]
    faskes = pd.read_excel(FILE_FASKES, dtype=str).iloc[:39]

    miskin = miskin[miskin["Provinsi"] != "Indonesia"]
    faskes = faskes[faskes["Provinsi"] != "Indonesia"]

    miskin = miskin.rename(columns={
        "Provinsi": "provinsi",
        "Garis Kemiskinan - September (Rp)": "garis_kemiskinan",
        "Jumlah Penduduk Miskin - September (ribu) (Ribu)": "jumlah_miskin_ribu",
        "Persentase Penduduk Miskin - September": "persen_miskin",
    })[["provinsi", "garis_kemiskinan", "jumlah_miskin_ribu", "persen_miskin"]]

    prefix = "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - "
    faskes = faskes.rename(columns={
        "Provinsi": "provinsi",
        prefix + "Rumah Sakit": "desa_rs",
        prefix + "Puskesmas": "desa_puskesmas",
        prefix + "Puskesmas Pembantu": "desa_pustu",
        prefix + "Apotek": "desa_apotek",
    })[["provinsi", "desa_rs", "desa_puskesmas", "desa_pustu", "desa_apotek"]]

    miskin["provinsi"] = miskin["provinsi"].str.strip()
    faskes["provinsi"] = faskes["provinsi"].str.strip()

    # Simbol strip BPS artinya tidak ada, bukan hilang acak, jadi jadi 0.
    faskes[FEATS] = faskes[FEATS].replace({"–": 0, "-": 0})
    faskes[FEATS] = faskes[FEATS].apply(pd.to_numeric, errors="raise").astype(int)

    for kolom in ["garis_kemiskinan", "jumlah_miskin_ribu", "persen_miskin"]:
        miskin[kolom] = pd.to_numeric(miskin[kolom], errors="raise")

    df = miskin.merge(faskes, on="provinsi", how="inner", validate="one_to_one")
    return df


def latih_model_final(df):
    """Latih 1 model final di semua 38 provinsi, untuk halaman Prediksi.

    Ini beda dengan evaluasi LOOCV di halaman Pemodelan. LOOCV itu untuk
    menguji (tiap provinsi ditahan bergiliran). Model final ini untuk
    memakai: dilatih sekali di semua data, lalu dipakai menebak input baru.
    """
    X = df[FEATS].values
    y = df[TARGET].values
    ridge_final = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X, y)
    rf_final = RandomForestRegressor(
        n_estimators=200, max_depth=3, min_samples_leaf=3, random_state=42
    ).fit(X, y)
    return ridge_final, rf_final


def hitung_metrik(y_asli, y_pred):
    mae = mean_absolute_error(y_asli, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_asli, y_pred)))
    r2 = r2_score(y_asli, y_pred)
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Tampilan utama
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Prediksi Kemiskinan Provinsi", layout="wide")
st.title("Analisis dan Prediksi Tingkat Kemiskinan Provinsi")
st.write(
    "Berdasarkan ketersediaan sarana kesehatan desa/kelurahan. "
    "Data: BPS, kemiskinan September 2025 dan sarana kesehatan 2024, 38 provinsi."
)

# Kalau file Excel belum ada di folder yang sama, beri tahu dengan jelas,
# jangan biarkan aplikasi crash tanpa penjelasan.
if not (os.path.exists(FILE_MISKIN) and os.path.exists(FILE_FASKES)):
    st.error(
        "File Excel belum ditemukan di folder yang sama dengan app.py. "
        "Salin 2 file ini ke folder yang sama dengan app.py, lalu jalankan ulang:\n"
        f"- {FILE_MISKIN}\n"
        f"- {FILE_FASKES}"
    )
    st.stop()

df = load_data()
X = df[FEATS].values
y = df[TARGET].values

# Menu di kiri, persis 4 halaman di Bab 7.4 laporan.
halaman = st.sidebar.radio(
    "Pilih halaman",
    ["Dashboard", "Data & Normalisasi", "Pemodelan", "Prediksi"],
)

# ---------------------------------------------------------------------------
# Halaman 1: Dashboard
# ---------------------------------------------------------------------------
if halaman == "Dashboard":
    st.header("Dashboard")
    st.write(
        "Ringkasan cepat: kemiskinan itu timpang antarwilayah, "
        "bukan sekadar rata-rata nasional."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provinsi", "38")
    col2.metric("Median kemiskinan", f"{df[TARGET].median():.3f}%")
    col3.metric("Terendah (Bali)", f"{df[TARGET].min():.2f}%")
    col4.metric("Tertinggi (Papua Tengah)", f"{df[TARGET].max():.2f}%")

    st.subheader("Top 10 provinsi dengan kemiskinan tertinggi")
    top10 = df.nlargest(10, TARGET).sort_values(TARGET)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10["provinsi"], top10[TARGET])
    ax.set_xlabel("Persen miskin (%)")
    st.pyplot(fig)

    st.subheader("Distribusi persen miskin")
    st.write(
        "Rata-rata (10,27%) lebih tinggi dari median (9,345%). "
        "Artinya distribusi miring kanan, ditarik oleh Papua."
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET], bins=10)
    ax.axvline(df[TARGET].median(), linestyle="--", label="Median 9,345%")
    ax.axvline(df[TARGET].mean(), linestyle=":", label="Mean 10,27%")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Halaman 2: Data & Normalisasi
# ---------------------------------------------------------------------------
elif halaman == "Data & Normalisasi":
    st.header("Data & Normalisasi")
    st.write(
        "Data gabungan 38 provinsi x 8 kolom: 1 kunci provinsi, "
        "6 fitur asli, dan 1 target persen_miskin. Nilai kosong: 0."
    )
    st.dataframe(df, use_container_width=True)

    st.subheader("Statistik deskriptif")
    st.dataframe(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    st.subheader("Contoh normalisasi (z-score)")
    st.write(
        "Rumus z-score: (nilai - rata-rata) / simpangan baku. "
        "Tujuannya menyamakan skala, supaya variabel yang angkanya besar "
        "(misalnya apotek sampai 2.903) tidak mendominasi model linear. "
        "Di halaman Pemodelan, Ridge memakai scaler di dalam pipeline, "
        "dan scaler itu hanya dipasang di data latih tiap fold, "
        "supaya data uji tidak bocor ke proses normalisasi. "
        "Tabel di bawah ini hanya contoh tampilan z-score dari semua data."
    )
    scaler_tampil = StandardScaler()
    df_z = df.copy()
    df_z[FEATS] = scaler_tampil.fit_transform(df[FEATS])
    st.dataframe(df_z[["provinsi"] + FEATS], use_container_width=True)

    st.subheader("Validasi cleaning")
    st.write(
        f"Total desa ber-RS: {int(df['desa_rs'].sum())} | "
        f"Puskesmas: {int(df['desa_puskesmas'].sum())} | "
        f"Pustu: {int(df['desa_pustu'].sum())} | "
        f"Apotek: {int(df['desa_apotek'].sum())}. "
        "Total ini sama dengan baris Indonesia di file asli, "
        "jadi cleaning tidak menghilangkan data."
    )

# ---------------------------------------------------------------------------
# Halaman 3: Pemodelan
# ---------------------------------------------------------------------------
elif halaman == "Pemodelan":
    st.header("Pemodelan")
    st.write(
        "Dua model: Ridge Regression dan Random Forest Regressor, "
        "dievaluasi dengan LOOCV (Leave-One-Out Cross-Validation). "
        "LOOCV artinya: tahan 1 provinsi jadi uji, latih di 37 sisanya, "
        "ulangi 38 kali. Cara ini dipilih karena data hanya 38 provinsi."
    )
    st.write(
        "Fitur model hanya 4 sarana kesehatan. Garis kemiskinan dan jumlah "
        "penduduk miskin tidak dipakai di model, karena turunan dari target."
    )

    if st.button("Hitung evaluasi LOOCV"):
        with st.spinner("Menghitung 38 fold LOOCV, tunggu sebentar..."):
            loo = LeaveOneOut()
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=10))
            pred_ridge = cross_val_predict(ridge, X, y, cv=loo)

            rf = RandomForestRegressor(
                n_estimators=200, max_depth=3, min_samples_leaf=3, random_state=42
            )
            pred_rf = cross_val_predict(rf, X, y, cv=loo)

            pred_base = []
            for latih, _uji in loo.split(X):
                pred_base.append(y[latih].mean())
            pred_base = np.array(pred_base)

            hasil = []
            for nama, pred in [
                ("Ridge", pred_ridge),
                ("Random Forest", pred_rf),
                ("Baseline rata-rata", pred_base),
            ]:
                mae, rmse, r2 = hitung_metrik(y, pred)
                hasil.append({"Model": nama, "MAE": round(mae, 4),
                              "RMSE": round(rmse, 4), "R2": round(r2, 4)})
            st.dataframe(pd.DataFrame(hasil), use_container_width=True)
            st.write(
                "Cara baca: MAE dan RMSE makin kecil makin baik, dalam poin persen. "
                "R2 = 1 sempurna, 0 sama dengan menebak rata-rata, "
                "negatif berarti lebih buruk dari rata-rata. "
                "Random Forest adalah yang terbaik, Ridge kalah dari baseline."
            )

            # Tabel kesalahan terbesar + grafik aktual vs prediksi
            err = pd.DataFrame({
                "provinsi": df["provinsi"].values,
                "aktual": y,
                "prediksi_rf": pred_rf,
            })
            err["galat"] = (err["aktual"] - err["prediksi_rf"]).abs()
            st.subheader("8 kesalahan terbesar (Random Forest)")
            st.dataframe(
                err.sort_values("galat", ascending=False).head(8).round(2),
                use_container_width=True,
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y, pred_rf)
            ax.plot([0, 30], [0, 30], linestyle="--")
            ax.set_xlabel("Aktual (%)")
            ax.set_ylabel("Prediksi Random Forest (%)")
            ax.set_title("Aktual vs Prediksi (garis putus-putus = prediksi sempurna)")
            st.pyplot(fig)

            st.subheader("Feature importance (Random Forest, latih di semua data)")
            _ridge_final, rf_final = latih_model_final(df)
            imp = sorted(
                zip(FEATS, rf_final.feature_importances_), key=lambda t: -t[1]
            )
            st.dataframe(
                pd.DataFrame(imp, columns=["Variabel", "Importance"]).round(4),
                use_container_width=True,
            )
            st.write(
                "Apotek paling dominan (sekitar 0,75). Hati-hati: ini bukan bukti "
                "apotek menyebabkan kemiskinan turun. Kemungkinan besar apotek "
                "hanya proksi ukuran provinsi (Jawa vs Papua)."
            )
    else:
        st.info("Klik tombol di atas untuk menghitung evaluasi LOOCV.")

# ---------------------------------------------------------------------------
# Halaman 4: Prediksi
# ---------------------------------------------------------------------------
else:
    st.header("Prediksi")
    st.write(
        "Isi 4 angka sarana kesehatan, lalu klik Prediksi. "
        "Model yang dipakai untuk prediksi utama adalah Random Forest, "
        "karena MAE-nya paling kecil di evaluasi LOOCV."
    )

    # Nilai awal = median tiap variabel (provinsi tipikal).
    rs = st.number_input("Desa/kelurahan ber-Rumah Sakit", min_value=9, max_value=346, value=42)
    puskesmas = st.number_input("Desa/kelurahan ber-Puskesmas", min_value=52, max_value=1110, value=208)
    pustu = st.number_input("Desa/kelurahan ber-Puskesmas Pembantu", min_value=147, max_value=2218, value=442)
    apotek = st.number_input("Desa/kelurahan ber-Apotek", min_value=8, max_value=2903, value=262)

    st.write(
        "Batas input di atas adalah min-max data asli. "
        "Random Forest tidak bisa menebak dengan baik di luar rentang "
        "yang pernah dilihatnya, jadi jangan isi jauh di luar batas itu."
    )

    if st.button("Prediksi"):
        _ridge_final, rf_final = latih_model_final(df)
        # Urutan input harus sama persis dengan FEATS.
        data_baru = [[rs, puskesmas, pustu, apotek]]
        p_rf = float(rf_final.predict(data_baru)[0])
        p_ridge = float(_ridge_final.predict(data_baru)[0])

        st.success(f"Prediksi Random Forest: {p_rf:.2f}%")
        st.write(f"Prediksi Ridge (pembanding): {p_ridge:.2f}%")
        st.warning(
            "Prediksi ini indikatif saja, bukan dasar anggaran. "
            "MAE Random Forest sekitar 3,87 poin, jadi nilai sebenarnya "
            "bisa meleset beberapa poin dari angka di atas. "
            "Hasil ini asosiasi, bukan sebab-akibat."
        )

    st.subheader("Contoh cepat untuk demo")
    st.write(
        "- Nilai median di atas: prediksi sekitar 9-10% (provinsi tipikal).\n"
        "- Papua Tengah (RS 14, Puskesmas 98, Pustu 148, Apotek 45): "
        "aslinya 29,45%, tapi model kemungkinan hanya memprediksi sekitar 17%. "
        "Ini contoh keterbatasan: model meremehkan provinsi ekstrem."
    )
