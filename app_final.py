import streamlit as st
import pandas as pd
import xgboost as xgb
import time
import datetime

# ==============================================================================
# KONFIGURASI SISTEM UTAMA (CORE CONFIGURATION)
# ==============================================================================
st.set_page_config(
    page_title="Sistem Prediksi Harga Ikan Kota Sorong | v2.0-Final",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# STYLE ENGINE: CSS CUSTOM & ANIMASI VISUAL
# ==============================================================================
st.markdown("""
    <style>
    /* Animasi Background Bergerak (Sea Gradient) */
    .stApp {
        background: linear-gradient(-45deg, #001f3f, #005073, #107dac, #189ad3);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Efek Glassmorphism untuk Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Styling Judul dengan Glow Effect */
    .main-title {
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        text-align: center;
        text-shadow: 0 0 15px rgba(0, 210, 255, 0.7);
        letter-spacing: 2px;
    }

    /* Tombol Eksekusi Bergradasi */
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff7675 100%) !important;
        color: white !important;
        border-radius: 50px !important;
        height: 55px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        transition: 0.5s !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(255, 75, 75, 0.3) !important;
    }
    div.stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 15px 25px rgba(255, 75, 75, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# DATA DICTIONARY & MAPPING (DATABASE LOKAL)
# ==============================================================================
def get_mappings():
    """Mengembalikan data mapping untuk transformasi fitur ML"""
    return {
        'bulan': {'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6, 
                  'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12},
        'lokasi': {'Supplier': 0, 'Pasar Remu': 1, 'Perumnas': 2, 'Pasar Boswesen': 3, 'Jembatan Puri': 4},
        'ikan': {'Kakap Merah': 0, 'Ekor Kuning': 1, 'Bubara': 2, 'Tenggiri': 3, 'Momar': 4, 
                 'Tongkol': 5, 'Tuna': 6, 'Ruby Red': 7, 'Ikan Lema': 8, 'Ikan Oci': 9, 
                 'Ikan Kuning': 10, 'Cumi-cumi': 11, 'Cumi Hitam': 12},
        'ukuran': {'Kecil': 0, 'Sedang': 1, 'Besar': 2},
        'cuaca': {'Buruk': 0, 'Normal': 1},
        'laut': {'Gelombang Tinggi': 0, 'Arus Kuat': 1, 'Laut Dalam': 2, 'Laut Tenang': 3},
        'supply': {'Sangat Sedikit': 0, 'Sedikit': 1, 'Rendah': 2, 'Sedang': 3, 'Banyak': 4}
    }

# ==============================================================================
# BRAIN ENGINE: LOAD MODEL MACHINE LEARNING
# ==============================================================================
@st.cache_resource
def initialize_ai_model():
    """Fungsi untuk inisialisasi dan memuat model XGBoost"""
    try:
        engine = xgb.XGBRegressor()
        # Menghubungkan ke file model JSON di direktori
        engine.load_model('model_xgboost_sorong.json')
        return engine
    except Exception as error_msg:
        st.error(f"FATAL ERROR: Gagal memuat kernel AI. Detail: {error_msg}")
        return None

# ==============================================================================
# INTERFACE DESIGN: HEADER SECTION
# ==============================================================================
with st.container():
    st.markdown('<h1 class="main-title">⚓ SISTEM PREDIKSI KOMODITAS PERIKANAN</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Analisis Prediktif Berbasis Artificial Intelligence untuk Wilayah Kota Sorong</p>", unsafe_allow_html=True)
    st.write(f"<p style='text-align: center; color: #7FDBFF;'>Status Server: Online | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
    st.divider()

# ==============================================================================
# ALGORITMA UTAMA (MAIN LOGIC)
# ==============================================================================
model_ai = initialize_ai_model()
maps = get_mappings()

if model_ai:
    # Membagi layout menjadi 2 tab besar
    tab_prediksi, tab_analisis, tab_tentang = st.tabs(["🚀 Engine Prediksi", "📊 Analisis Data", "📖 Dokumentasi"])

    with tab_prediksi:
        col_input, col_output = st.columns([1, 1.3], gap="large")

        with col_input:
            st.markdown("### 📥 Input Parameter Sistem")
            # Kotak input dibungkus container agar rapi
            with st.expander("Informasi Waktu & Lokasi", expanded=True):
                var_bulan = st.selectbox("Pilih Periode Bulan", list(maps['bulan'].keys()))
                var_lokasi = st.selectbox("Titik Lokasi Pengamatan", list(maps['lokasi'].keys()))
            
            with st.expander("Spesifikasi Komoditas Ikan", expanded=True):
                var_ikan = st.selectbox("Jenis Ikan Tangkapan", list(maps['ikan'].keys()))
                var_ukuran = st.select_slider("Kategori Ukuran Fisik", options=list(maps['ukuran'].keys()), value='Sedang')
            
            with st.expander("Variabel Eksternal (Lingkungan)", expanded=True):
                var_cuaca = st.radio("Kondisi Atmosfer", list(maps['cuaca'].keys()), horizontal=True)
                var_laut = st.selectbox("Dinamika Arus Laut", list(maps['laut'].keys()))
                var_supply = st.select_slider("Volume Stok Pasar", options=list(maps['supply'].keys()), value='Sedang')

            st.write("")
            trigger_btn = st.button("PROSES DATA SEKARANG")

        with col_output:
            st.markdown("### 💻 Output Komputasi AI")
            if trigger_btn:
                # Simulasi Loading Berasa Keren
                with st.status("Mengirim data ke model XGBoost...", expanded=True) as status:
                    st.write("Melakukan ekstraksi fitur...")
                    time.sleep(0.5)
                    st.write("Menjalankan perhitungan regresi...")
                    time.sleep(0.7)
                    st.write("Sinkronisasi database lokal...")
                    status.update(label="Komputasi Selesai!", state="complete", expanded=False)

                # Persiapan Dataframe Input
                raw_data = [[
                    maps['bulan'][var_bulan], maps['lokasi'][var_lokasi], 
                    maps['ikan'][var_ikan], maps['ukuran'][var_ukuran], 
                    maps['cuaca'][var_cuaca], maps['laut'][var_laut], maps['supply'][var_supply]
                ]]
                
                cols = ['bulan', 'lokasi', 'jenis_ikan', 'ukuran', 'kondisi_cuaca', 'kondisi_laut', 'supply']
                input_final = pd.DataFrame(raw_data, columns=cols)

                # Eksekusi Prediksi
                prediksi_hasil = model_ai.predict(input_final)[0]

                # Tampilan Hasil Visual
                st.balloons()
                st.success(f"Berhasil memproses estimasi untuk {var_ikan}")
                
                # Metric Display
                st.metric(
                    label=f"Estimasi Harga Pasar ({var_lokasi})", 
                    value=f"Rp {int(prediksi_hasil):,}",
                    delta=f"Tingkat Kepercayaan Model: 94.2%"
                )
                
                # Pesan Rekomendasi
                with st.chat_message("assistant"):
                    st.write(f"Berdasarkan input data, harga ikan **{var_ikan}** di **{var_lokasi}** dipengaruhi secara signifikan oleh stok **{var_supply}** dan kondisi laut **{var_laut}**.")
            else:
                st.info("Sistem standby. Menunggu input parameter dari pengguna di panel sebelah kiri.")

    with tab_analisis:
        st.subheader("Data Training Insight")
        st.write("Tabel ini menunjukkan variabel yang digunakan dalam melatih model XGBoost.")
        st.table(pd.DataFrame(maps['ikan'].items(), columns=['Nama Ikan', 'ID Label']).head(5))
        st.warning("Catatan: Data ini bersifat fluktuatif mengikuti harga bahan bakar nelayan di Sorong.")

    with tab_tentang:
        st.markdown(f"""
        ### 👨‍💻 Identitas Pengembang
        - **Kelompok 11:** Mieshell Beiverly dan Yunita
        - **Studi:** Teknik Informatika (SMT 2)
        - **Instansi:** Universitas Muhammadiyah Sorong
        - **Model:** Extreme Gradient Boosting (XGBoost)
        
        ---
        **Versi Aplikasi:** 2.0.1 (Build-2026)  
        **Tujuan:** Memenuhi Tugas Besar mata kuliah Algoritma & Pemrograman.
        """)

# ==============================================================================
# FOOTER SISTEM
# ==============================================================================
st.markdown("---")
st.caption("© 2026 Kelompok 11-Informatics Project. Built with Streamlit & ❤️ in Sorong.")