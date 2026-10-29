"""
Penebak Persen Kemiskinan Provinsi.
Isi 4 angka fasilitas kesehatan desa, lihat tebakan persen penduduk miskin.
Data BPS: kemiskinan September 2025 dan sarana kesehatan 2024.
Model hanya memakai 4 angka sarana. Hasilnya cuma tebakan, bukan angka resmi.
Semua angka ringkasan (median, rata-rata, MAE, contoh) dihitung dari data,
bukan ditulis manual, supaya tidak basi kalau file Excel diganti.
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

# Nama kolom versi manusia, dipakai di label dan tabel.
LABEL = {
    "desa_rs": "Rumah Sakit",
    "desa_puskesmas": "Puskesmas",
    "desa_pustu": "Puskesmas Pembantu",
    "desa_apotek": "Apotek",
}

N_TERBESAR = 8  # jumlah baris di tabel "selisih terbesar"
ALPHA = 10  # kekuatan penalti Ridge
NAMA_RF = "Random Forest"
NAMA_RIDGE = "Ridge"
NAMA_BASE = "Baseline rata-rata"


def _pisah_baris_indonesia(mentah):
    """Potong tabel BPS jadi (baris provinsi, baris Indonesia).

    File BPS menaruh 38 provinsi lebih dulu, lalu baris "Indonesia", lalu
    catatan kaki. Memotong tepat di nama "Indonesia" lebih tahan kalau
    jumlah provinsi berubah, dibanding menebak "39 baris pertama".
    """
    provinsi = mentah["Provinsi"].astype("string").str.strip()
    if "Indonesia" not in provinsi.tolist():
        raise ValueError(
            "Baris 'Indonesia' tidak ditemukan. File BPS sepertinya berubah format."
        )
    posisi = provinsi.tolist().index("Indonesia")
    data = mentah.iloc[:posisi]
    if data["Provinsi"].isna().any():
        raise ValueError("Ada baris tanpa nama provinsi di atas baris 'Indonesia'.")
    return data, mentah.iloc[posisi]


def _ke_angka(nilai):
    """Ubah satu sel BPS jadi angka. Strip berarti tidak ada, jadi 0."""
    teks = str(nilai).strip()
    if teks in {"–", "-", "nan"}:
        return 0
    return pd.to_numeric(teks, errors="raise")


def _stempel_file():
    """Waktu ubah kedua file Excel, dipakai sebagai kunci cache.

    Kalau file diganti, cache ikut batal, jadi angka di layar tidak basi.
    """
    return tuple(os.path.getmtime(f) for f in (FILE_MISKIN, FILE_FASKES))


@st.cache_data(show_spinner="Membaca file Excel BPS...")
def load_data(stempel):
    """Baca, bersihkan, dan gabungkan 2 file Excel BPS (Bab 3 laporan).

    Langkahnya sama dengan yang sudah diverifikasi di Colab:
    baca sebagai teks, ambil baris provinsi saja, pilih September 2025 dan
    4 sarana, ubah nama kolom jadi snake_case, ubah teks jadi angka, lalu
    inner join. Sekalian dicek apakah total tiap sarana sama dengan baris
    Indonesia di file asli.
    """
    mentah_miskin = pd.read_excel(FILE_MISKIN, dtype=str)
    mentah_faskes = pd.read_excel(FILE_FASKES, dtype=str)

    miskin, _ = _pisah_baris_indonesia(mentah_miskin)
    faskes, total_indonesia = _pisah_baris_indonesia(mentah_faskes)

    miskin = miskin.rename(columns={
        "Provinsi": "provinsi",
        "Garis Kemiskinan - September (Rp)": "garis_kemiskinan",
        "Jumlah Penduduk Miskin - September (ribu) (Ribu)": "jumlah_miskin_ribu",
        "Persentase Penduduk Miskin - September": "persen_miskin",
    })[["provinsi", "garis_kemiskinan", "jumlah_miskin_ribu", "persen_miskin"]]

    prefix = "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - "
    kolom_faskes = {
        "Provinsi": "provinsi",
        prefix + "Rumah Sakit": "desa_rs",
        prefix + "Puskesmas": "desa_puskesmas",
        prefix + "Puskesmas Pembantu": "desa_pustu",
        prefix + "Apotek": "desa_apotek",
    }
    faskes = faskes.rename(columns=kolom_faskes)
    faskes = faskes[["provinsi", "desa_rs", "desa_puskesmas", "desa_pustu", "desa_apotek"]]

    miskin["provinsi"] = miskin["provinsi"].str.strip()
    faskes["provinsi"] = faskes["provinsi"].str.strip()

    # Simbol strip BPS artinya tidak ada, bukan hilang acak, jadi jadi 0.
    faskes[FEATS] = faskes[FEATS].replace({"–": 0, "-": 0})
    faskes[FEATS] = faskes[FEATS].apply(pd.to_numeric, errors="raise").astype(int)

    for kolom in ["garis_kemiskinan", "jumlah_miskin_ribu", "persen_miskin"]:
        miskin[kolom] = pd.to_numeric(miskin[kolom], errors="raise")

    df = miskin.merge(faskes, on="provinsi", how="inner", validate="one_to_one")

    # Baris Indonesia = jumlah seluruh provinsi menurut BPS. Cocokkan.
    total_resmi = {
        kolom: int(_ke_angka(total_indonesia[kolom_asli]))
        for kolom_asli, kolom in kolom_faskes.items()
        if kolom in FEATS
    }
    return df, total_resmi


def latih_model_final(df):
    """Latih 1 model final di semua provinsi, untuk halaman Prediksi.

    Ini beda dengan evaluasi LOOCV di halaman Pemodelan. LOOCV itu untuk
    menguji (tiap provinsi ditahan bergiliran). Model final ini untuk
    memakai: dilatih sekali di semua data, lalu dipakai menebak input baru.
    """
    X = df[FEATS].to_numpy()
    y = df[TARGET].to_numpy()
    ridge_final = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(X, y)
    rf_final = RandomForestRegressor(
        n_estimators=200, max_depth=3, min_samples_leaf=3, random_state=42
    ).fit(X, y)
    return ridge_final, rf_final


def hitung_metrik(y_asli, y_pred):
    mae = mean_absolute_error(y_asli, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_asli, y_pred)))
    r2 = r2_score(y_asli, y_pred)
    return mae, rmse, r2


@st.cache_data(show_spinner=False)
def evaluasi_loocv(df):
    """Uji 2 model + baseline dengan LOOCV, plus pengaruh tiap fasilitas.

    Mengembalikan 3 tabel: metrik, selisih per provinsi, dan feature
    importance. Hasilnya di-cache supaya tidak dihitung dua kali saat
    dipakai di halaman Pemodelan dan Prediksi.
    """
    X = df[FEATS].to_numpy()
    y = df[TARGET].to_numpy()
    loo = LeaveOneOut()

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=3, min_samples_leaf=3, random_state=42
    )
    tebakan = {
        NAMA_RIDGE: cross_val_predict(ridge, X, y, cv=loo),
        NAMA_RF: cross_val_predict(rf, X, y, cv=loo),
        NAMA_BASE: np.array([y[latih].mean() for latih, _uji in loo.split(X)]),
    }

    metrik = []
    for nama, pred in tebakan.items():
        mae, rmse, r2 = hitung_metrik(y, pred)
        metrik.append({"Model": nama, "MAE": mae, "RMSE": rmse, "R2": r2})
    metrik = pd.DataFrame(metrik)

    selisih = pd.DataFrame({
        "provinsi": df["provinsi"].to_numpy(),
        "aktual": y,
        "prediksi": tebakan[NAMA_RF],
    })
    selisih["selisih"] = (selisih["aktual"] - selisih["prediksi"]).abs()

    _ridge_final, rf_final = latih_model_final(df)
    pengaruh = pd.DataFrame({
        "Fasilitas": [LABEL[kolom] for kolom in FEATS],
        "Pengaruh": rf_final.feature_importances_,
    }).sort_values("Pengaruh", ascending=False, ignore_index=True)

    return metrik, selisih, pengaruh


# ---------------------------------------------------------------------------
# Tampilan utama
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Penebak Persen Kemiskinan", layout="wide")
st.title("Penebak Persen Kemiskinan")
st.write(
    "Isi 4 angka fasilitas kesehatan desa, lihat tebakan persen penduduk miskin. "
    "Data BPS: kemiskinan September 2025 dan sarana kesehatan 2024."
)

# Kalau file Excel belum ada di folder yang sama, beri tahu dengan jelas,
# jangan biarkan aplikasi crash tanpa penjelasan.
if not (os.path.exists(FILE_MISKIN) and os.path.exists(FILE_FASKES)):
    st.error(
        "2 file Excel belum ditemukan di folder ini. "
        "Salin kedua file Excel BPS ke folder yang sama dengan app.py, lalu jalankan ulang."
    )
    st.stop()

df, total_resmi = load_data(_stempel_file())
X = df[FEATS].to_numpy()
y = df[TARGET].to_numpy()

provinsi_termiskin = df.loc[df[TARGET].idxmax()]
provinsi_terkaya = df.loc[df[TARGET].idxmin()]
rata_rata = float(df[TARGET].mean())
nilai_tengah = float(df[TARGET].median())

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
    st.write(f"Gambaran cepat {len(df)} provinsi.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provinsi", len(df))
    col2.metric("Nilai tengah", f"{nilai_tengah:.2f}%")
    col3.metric(f"Terendah: {provinsi_terkaya['provinsi']}", f"{provinsi_terkaya[TARGET]:.2f}%")
    col4.metric(f"Tertinggi: {provinsi_termiskin['provinsi']}", f"{provinsi_termiskin[TARGET]:.2f}%")

    st.subheader("10 provinsi termiskin")
    top10 = df.nlargest(10, TARGET).sort_values(TARGET)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10["provinsi"], top10[TARGET])
    ax.set_xlabel("Persen miskin (%)")
    ax.set_title(f"10 provinsi termiskin ({top10[TARGET].min():.2f}–{top10[TARGET].max():.2f}%)")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Distribusi persen miskin")
    arah = "lebih tinggi" if rata_rata > nilai_tengah else "lebih rendah"
    pesan = (
        f"Rata-rata {rata_rata:.2f}% {arah} dari nilai tengah {nilai_tengah:.2f}%."
    )
    if rata_rata > nilai_tengah:
        pesan += (
            " Artinya ada beberapa provinsi yang kemiskinannya jauh di atas "
            "kebanyakan, sehingga rata-rata ikut tertarik naik."
        )
    else:
        pesan += " Artinya sebarannya relatif mendekati simetris."
    st.write(pesan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET], bins=10)
    ax.axvline(nilai_tengah, linestyle="--", label=f"Nilai tengah {nilai_tengah:.2f}%")
    ax.axvline(rata_rata, linestyle=":", label=f"Rata-rata {rata_rata:.2f}%")
    ax.set_xlabel("Persen miskin (%)")
    ax.set_ylabel("Jumlah provinsi")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Halaman 2: Data & Normalisasi
# ---------------------------------------------------------------------------
elif halaman == "Data & Normalisasi":
    st.header("Data & Normalisasi")
    st.write(
        f"Tabel gabungan {len(df)} provinsi. Kolom persen_miskin adalah angka yang ditebak model. "
        "Tanda strip di file BPS (artinya sarana tidak tersedia) sudah diubah jadi 0."
    )
    st.dataframe(df)

    st.subheader("Ringkasan tiap kolom")
    st.write("Rata-rata, terendah, tertinggi, dan nilai tengah tiap kolom.")
    st.dataframe(df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    st.subheader("Contoh penyamaan skala")
    st.write(
        "Supaya angka besar dan kecil bisa dibandingkan, tiap angka diubah "
        "jadi selisihnya dari rata-rata. Tabel di bawah ini cuma contoh tampilan, "
        "bukan cara model menghitung di dalam."
    )
    scaler_tampil = StandardScaler()
    df_z = df.copy()
    df_z[FEATS] = scaler_tampil.fit_transform(df[FEATS])
    st.dataframe(df_z[["provinsi"] + FEATS])

    st.subheader("Cek total data")
    total_gabungan = {kolom: int(df[kolom].sum()) for kolom in FEATS}
    rincian = " | ".join(
        f"{LABEL[kolom]}: {total_gabungan[kolom]:,}".replace(",", ".")
        for kolom in FEATS
    )
    st.write(f"Total desa menurut tabel gabungan: {rincian}.")

    selisih_total = {
        kolom: total_gabungan[kolom] - total_resmi[kolom] for kolom in FEATS
    }
    if all(nilai == 0 for nilai in selisih_total.values()):
        st.success(
            "Total tiap sarana sama dengan baris Indonesia di file asli, "
            "jadi tidak ada data yang hilang saat digabung."
        )
    else:
        beda = ", ".join(
            f"{LABEL[kolom]} {selisih_total[kolom]:+d}" for kolom in FEATS
        )
        st.warning(
            f"Total tabel gabungan beda dengan baris Indonesia di file asli ({beda}). "
            "Ada baris provinsi yang tidak ikut tergabung."
        )

# ---------------------------------------------------------------------------
# Halaman 3: Pemodelan
# ---------------------------------------------------------------------------
elif halaman == "Pemodelan":
    st.header("Pemodelan")
    st.write(
        "2 model diuji dengan cara bergilir: 1 provinsi disembunyikan jadi soal, "
        f"{len(df) - 1} provinsi sisanya dipakai belajar, diulang {len(df)} kali. "
        "Cara ini disebut LOOCV."
    )
    st.write(
        "Model hanya memakai 4 angka sarana kesehatan. "
        "Garis kemiskinan dan jumlah penduduk miskin tidak dipakai, "
        "karena keduanya sudah turunan dari angka yang ditebak."
    )

    if st.button("Hitung hasil uji (LOOCV)"):
        with st.spinner("Menghitung, tunggu sebentar..."):
            metrik, selisih, pengaruh = evaluasi_loocv(df)
            st.dataframe(metrik.round(4))

            terbaik = metrik.loc[metrik["MAE"].idxmin()]
            st.write(
                f"MAE terkecil: {terbaik['Model']} ({terbaik['MAE']:.2f} poin persen). "
                "MAE adalah rata-rata selisih tebakan, jadi semakin kecil semakin bagus."
            )

            mae_ridge = float(metrik.loc[metrik["Model"] == NAMA_RIDGE, "MAE"].iloc[0])
            mae_base = float(metrik.loc[metrik["Model"] == NAMA_BASE, "MAE"].iloc[0])
            if mae_ridge >= mae_base:
                st.info(
                    "Catatan jujur: Ridge belum mengalahkan baseline rata-rata. "
                    "Artinya 4 angka sarana ini belum punya pola garis lurus yang jelas "
                    "dengan kemiskinan. Yang menolong justru cara Random Forest "
                    "memotong data jadi kelompok."
                )

            st.subheader(f"{N_TERBESAR} selisih terbesar ({NAMA_RF})")
            st.dataframe(
                selisih.sort_values("selisih", ascending=False).head(N_TERBESAR).round(2)
            )

            batas_bawah = float(min(y.min(), selisih["prediksi"].min())) - 2
            batas_atas = float(max(y.max(), selisih["prediksi"].max())) + 2
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y, selisih["prediksi"])
            ax.plot([batas_bawah, batas_atas], [batas_bawah, batas_atas], linestyle="--")
            ax.set_xlim(batas_bawah, batas_atas)
            ax.set_ylim(batas_bawah, batas_atas)
            ax.set_xlabel("Aktual (%)")
            ax.set_ylabel(f"Prediksi {NAMA_RF} (%)")
            ax.set_title("Asli vs Tebakan")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Fasilitas mana yang paling berpengaruh?")
            st.dataframe(pengaruh.round(4))
            teratas = pengaruh.iloc[0]
            st.write(
                f"{teratas['Fasilitas']} paling besar pengaruhnya "
                f"({teratas['Pengaruh']:.1%} dari total). Ini bukan bukti "
                f"{teratas['Fasilitas'].lower()} menyebabkan kemiskinan turun. "
                "Bisa jadi karena provinsi besar memang punya banyak sarana."
            )
    else:
        st.info("Klik tombol di atas untuk menghitung hasil ujinya.")

# ---------------------------------------------------------------------------
# Halaman 4: Prediksi
# ---------------------------------------------------------------------------
else:
    st.header("Prediksi")
    st.write(
        "Isi 4 angka di bawah, lalu klik Prediksi. "
        f"Hasil utama memakai {NAMA_RF}, karena selisih rata-ratanya paling kecil."
    )

    ridge_final, rf_final = latih_model_final(df)

    # Nilai awal = median tiap variabel (provinsi tipikal).
    isian = {}
    for kolom in FEATS:
        isian[kolom] = st.number_input(
            f"Jumlah desa yang punya {LABEL[kolom]}",
            min_value=int(df[kolom].min()),
            max_value=int(df[kolom].max()),
            value=int(round(df[kolom].median())),
            step=1,
        )

    st.write(
        "Batas isian mengikuti data asli di atas. "
        "Di luar batas itu, tebakan jadi kurang bisa dipercaya."
    )

    if st.button("Prediksi"):
        # Urutan input harus sama persis dengan FEATS.
        data_baru = [[isian[kolom] for kolom in FEATS]]
        p_rf = float(rf_final.predict(data_baru)[0])
        p_ridge = float(ridge_final.predict(data_baru)[0])

        st.success(f"Prediksi {NAMA_RF}: {p_rf:.2f}%")
        st.write(f"Prediksi {NAMA_RIDGE} (pembanding): {p_ridge:.2f}%")

        with st.spinner("Mengecek rata-rata melesetnya..."):
            metrik, _selisih, _pengaruh = evaluasi_loocv(df)
        mae_rf = float(metrik.loc[metrik["Model"] == NAMA_RF, "MAE"].iloc[0])
        st.warning(
            f"Ini cuma tebakan. Saat diuji bergilir, tebakan {NAMA_RF} rata-rata "
            f"meleset {mae_rf:.1f} poin, jadi jangan dipakai sendirian untuk bagi anggaran."
        )

    st.subheader("Contoh cepat untuk demo")
    isian_tengah = [[int(round(df[kolom].median())) for kolom in FEATS]]
    p_tengah = float(rf_final.predict(isian_tengah)[0])
    st.write(
        f"- Isian median di atas (provinsi tipikal): tebakan sekitar **{p_tengah:.2f}%**. "
        "Ini tebakan untuk kombinasi angka tengah, bukan persentase rata-rata nasional."
    )

    p_ekstrem = float(rf_final.predict([provinsi_termiskin[FEATS].to_numpy()])[0])
    angka_ekstrem = " | ".join(
        f"{LABEL[kolom].replace('Puskesmas Pembantu', 'Pustu')} {provinsi_termiskin[kolom]}"
        for kolom in FEATS
    ).replace(" | ", ", ")
    arah = "meremehkan" if p_ekstrem < provinsi_termiskin[TARGET] else "melebih-lebihkan"
    st.write(
        f"- {provinsi_termiskin['provinsi']} ({angka_ekstrem}): aslinya "
        f"{provinsi_termiskin[TARGET]:.2f}%, model menebak {p_ekstrem:.2f}%. "
        f"Selisih {abs(provinsi_termiskin[TARGET] - p_ekstrem):.2f} poin: "
        f"model {arah} provinsi ekstrem."
    )
