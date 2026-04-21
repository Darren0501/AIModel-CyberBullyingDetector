import pandas as pd
import numpy as np
import re
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. KONFIGURASI 
# ==========================================
PAKAI_STEMMING = False #Stemming dimatikan karena cukup memakan waktu

FILENAME = 'DATASET_UTAMA.csv' 

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Sedang memuat data...")
try:
    df = pd.read_csv(FILENAME)
    df = df.dropna(subset=['text', 'Label_Final'])
    print(f"Total Data: {len(df)} baris")
except FileNotFoundError:
    print(f"ERROR: File '{FILENAME}' tidak ditemukan. Upload dulu ya!")
    df = pd.DataFrame()

# ==========================================
# 3. PREPROCESSING
# ==========================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def cleaning_optimal(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if PAKAI_STEMMING:
        text = stemmer.stem(text) 

    return text

if not df.empty:
    print(f"\nMelakukan Preprocessing (Stemming={PAKAI_STEMMING})...")
    df['text_clean'] = df['text'].apply(cleaning_optimal)
    print("Preprocessing Selesai!")

    # ==========================================
    # 4. SPLIT DATA
    # ==========================================
    X = df['text_clean']
    y = df['Label_Final']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 5. GRID SEARCH 
    # ==========================================
    print("\nSedang melakukan Tuning Otomatis (Grid Search)...")
    print("Komputer akan mencoba berbagai kombinasi settingan. Tunggu ya...")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC(class_weight='balanced')) # Balanced = Adil ke minoritas
    ])

    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_df': [0.75, 1.0],
        'clf__C': [0.1, 1, 10]
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # ==========================================
    # 6. HASIL & EVALUASI
    # ==========================================
    print("\nTUNING SELESAI!")
    print(f"Settingan Terbaik: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\nAkurasi di Data Test: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\n--- Laporan Klasifikasi Lengkap ---")
    print(classification_report(y_test, y_pred))

    # ==========================================
    # 7. SIMPAN MODEL JUARA
    # ==========================================
    joblib.dump(best_model, 'model_svm.pkl')
    print("Model terbaik disimpan sebagai 'model_svm.pkl'")
