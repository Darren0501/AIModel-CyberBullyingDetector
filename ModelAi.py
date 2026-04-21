import streamlit as st
import pandas as pd
import joblib
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV  # <--- WAJIB ADA
from sklearn.pipeline import Pipeline

MODEL_PATH = 'model_svm.pkl'
DATA_PATH = 'DATASET_UTAMA.csv'
FEEDBACK_PATH = 'temp_feedback.csv'

st.set_page_config(page_title="Sistem Deteksi Cyberbullying", layout="wide", page_icon="🛡️")


# BACKEND
def ensure_feedback_file():
    if not os.path.exists(FEEDBACK_PATH):
        df = pd.DataFrame(columns=['text', 'Label_Final'])
        df.to_csv(FEEDBACK_PATH, index=False, header=False)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            return None
    return None

def read_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        return pd.DataFrame(columns=['text', 'Label_Final'])
    try:
        df = pd.read_csv(FEEDBACK_PATH, header=None, names=['text', 'Label_Final'])
        df = df.dropna(subset=['text', 'Label_Final'])
        return df
    except:
        return pd.DataFrame(columns=['text', 'Label_Final'])

def simpan_feedback(text, label_benar):
    ensure_feedback_file()
    df = read_feedback()
    if text in df['text'].values:
        df.loc[df['text'] == text, 'Label_Final'] = label_benar
        df.to_csv(FEEDBACK_PATH, index=False, header=False)
    else:
        new_row = pd.DataFrame({'text': [text], 'Label_Final': [label_benar]})
        new_row.to_csv(FEEDBACK_PATH, mode='a', index=False, header=False)
    return True

# RETRAIN MODEL 
def retrain_model():
    progress_text = st.empty()
    bar = st.progress(0)
    progress_text.text("Memuat dataset...")

    # Load data utama
    try:
        if not os.path.exists(DATA_PATH):
            st.error("Dataset utama tidak ditemukan!")
            return
        df_main = pd.read_csv(DATA_PATH, on_bad_lines='skip')
        df_main = df_main[df_main['text'] != 'text']
        df_main = df_main[['text', 'Label_Final']]
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return

    bar.progress(20)

    # Load feedback
    df_feed = read_feedback()
    df_total = pd.concat([df_main, df_feed], ignore_index=True)
    df_total = df_total.drop_duplicates(subset=['text'], keep='last')
    df_total = df_total.dropna(subset=['text', 'Label_Final'])

    bar.progress(45)
    progress_text.text(f"🚀 Melatih ulang model...")

    # Training
    start = time.time()
    
    svm = LinearSVC(class_weight='balanced', C=10, dual=False)
    clf = CalibratedClassifierCV(svm, method='sigmoid') 

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ('clf', clf)
    ])

    X = df_total['text'].astype(str)
    y = df_total['Label_Final'].astype(str)

    try:
        pipeline.fit(X, y)
    except ValueError as e:
        st.error(f"Error Training: {e}. Pastikan minimal ada 2 kelas label berbeda.")
        return

    dur = time.time() - start
    bar.progress(90)

    # Simpan Model
    joblib.dump(pipeline, MODEL_PATH)
    df_total.to_csv(DATA_PATH, index=False)

    if os.path.exists(FEEDBACK_PATH):
        os.remove(FEEDBACK_PATH)

    bar.progress(100)
    progress_text.success(f"Training Selesai ({dur:.2f} detik).")
    st.cache_resource.clear()  
    time.sleep(1)
    st.rerun()

model = load_model()

# FRONTEND STREAMLIT
menu = st.sidebar.radio("Menu Navigasi", ["🏠 Simulasi Medsos", "⚙️ Admin & Training"])

# Cek Model 
if model is None:
    st.warning("⚠️ Model belum siap. Silakan klik tombol di bawah.")
    if st.button("🚀 Latih Model Pertama Kali"):
        retrain_model()
    st.stop()


# MENU 1: SIMULASI MEDSOS
if menu == "🏠 Simulasi Medsos":
    st.title("📱 Simulasi Media Sosial Aman")
    st.markdown("---")

    user_input = st.text_input("Tulis komentar...", placeholder="Ketik sesuatu...")

    if st.button("Kirim"):
        if user_input:
            try:
                pred_label = model.predict([user_input])[0]

                probs = model.predict_proba([user_input])[0]
                classes = model.classes_

                ranking = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                top_3 = ranking[:3]

                col_res1, col_res2 = st.columns([1.5, 1])
                
                with col_res1:
                    top_score = top_3[0][1] 
                    
                    if pred_label == 'Non-Bullying':
                        st.success(f"✅ **Terkirim!**")
                        st.write(f"**Anda:** {user_input}")
                    else:
                        st.error(f"**DIBLOKIR!**")
                        st.markdown(f"Terdeteksi sebagai: **{pred_label}** (Score: `{top_score:.4f}`)")
                        st.info("Komentar melanggar standar komunitas.")

                with col_res2:
                    st.write("**Top 3 Kategori**")
                    
                    df_top3 = pd.DataFrame(top_3, columns=['Kategori', 'Score'])
                    df_top3.set_index('Kategori', inplace=True)
                    
                    st.dataframe(
                        df_top3.style.background_gradient(cmap='Reds', vmin=0, vmax=1).format("{:.4f}"),
                        use_container_width=True
                    )

            except AttributeError:
                st.error("⚠️ Model usang terdeteksi. Silakan ke menu 'Admin' lalu klik 'Update Pengetahuan AI'.")

# MENU 2: ADMIN
elif menu == "⚙️ Admin & Training":
    st.title("🧠 Dashboard Pembelajaran AI")

    if 'prediksi_admin' not in st.session_state:
        st.session_state.prediksi_admin = None
    if 'text_admin' not in st.session_state:
        st.session_state.text_admin = ""
    if 'mode_koreksi' not in st.session_state:
        st.session_state.mode_koreksi = False

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("🛠️ Uji & Validasi")
        test_input = st.text_area("Masukkan kalimat tes:", height=80)

        if st.button("Analisis AI"):
            if test_input:
                res = model.predict([test_input])[0]
                st.session_state.prediksi_admin = res
                st.session_state.text_admin = test_input
                st.session_state.mode_koreksi = False
                
                st.info(f"Prediksi: **{res}**")

        if st.session_state.prediksi_admin:
            st.write("Apakah prediksi ini benar?")
            c1, c2 = st.columns(2)
            
            if c1.button("✅ Benar (Simpan)"):
                simpan_feedback(st.session_state.text_admin, st.session_state.prediksi_admin)
                st.success("Tersimpan!")
                time.sleep(1)
                st.session_state.prediksi_admin = None
                st.rerun()

            if c2.button("❌ Salah (Koreksi)"):
                st.session_state.mode_koreksi = True

            if st.session_state.mode_koreksi:
                opsi = list(model.classes_) if hasattr(model, 'classes_') else ['Non-Bullying', 'Makian', 'Hate Speech']
                label_koreksi = st.selectbox("Pilih kategori yang benar:", opsi)
                if st.button("Simpan Koreksi"):
                    simpan_feedback(st.session_state.text_admin, label_koreksi)
                    st.success(f"Tersimpan sebagai '{label_koreksi}'")
                    time.sleep(1)
                    st.session_state.prediksi_admin = None
                    st.session_state.mode_koreksi = False
                    st.rerun()

    with col2:
        st.subheader("📊 Status Data")
        df_feed = read_feedback()
        jumlah = len(df_feed)
        st.metric("Antrian Update", f"{jumlah} data")
        
        st.markdown("---")
        if st.button("🔄 Update Pengetahuan AI (Retrain)"):
            retrain_model()