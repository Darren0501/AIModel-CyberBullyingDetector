import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==========================================
# 1. LOAD SUMBER DAYA (KAMUS & DATASET)
# ==========================================
print("Memuat resources...")
try:
    kamus_alay = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
    alay_dict = dict(zip(kamus_alay[0], kamus_alay[1]))
    print("Kamus Alay dimuat.")
except:
    alay_dict = {}

try:
    df_abusive = pd.read_csv('abusive.csv')
    list_kasar = df_abusive.iloc[:, 0].astype(str).str.lower().tolist()
    set_kasar = set(list_kasar)
    print(f"Daftar Kata Kotor dimuat: {len(set_kasar)} kata.")
except:
    set_kasar = set()
    print("File abusive.csv tidak ditemukan! Filter Makian tidak akan maksimal.")

keywords = {
    'Pelecehan Seksual': ['bokep', 'sange', 'mesum', 'porno', 'lonte', 'perek', 'open bo', 'pap tt', 'tobrut', 'bugil', 'telanjang', 'ngaceng', 'coli', 'jablay', 'pelacur', 'perkosa', 'nude', 'seks', 'nenen', 'toket', 'pentil', 'selangkangan', 'onani', 'masturbasi', 'desah', 'wikwik', 'mantap-mantap', 'ani-ani', 'simpanan', 'germo', 'ayam kampus', 'bisyar'],
    'Ekonomi': ['miskin', 'kere', 'gembel', 'melarat', 'pengangguran', 'kismin', 'rakjel', 'bpjs', 'duafa', 'ngemis', 'minta-minta', 'ekonomi sulit', 'bansos', 'hp kentang', 'motor butut'],
    'Hate Speech / Ancaman': ['bunuh', 'mati', 'mampus', 'modar', 'nyawa', 'bacok', 'tebas', 'penggal', 'santet', 'habisi', 'lenyap', 'bakar', 'bom', 'halal darah', 'sampah masyarakat'],
    'Fisik': ['jelek', 'burik', 'item', 'dekil', 'gendut', 'kurus', 'botak', 'pesek', 'gigi', 'muka', 'kerempeng', 'gembrot', 'ceking', 'boncel', 'keling', 'daki', 'jerawat', 'ompong', 'tonggos']
}

# ==========================================
# 2. FUNGSI CLEANING & LABELING CERDAS
# ==========================================
def cleaning(text):
    text = str(text).lower()
    words = text.split()
    new_words = [alay_dict[w] if w in alay_dict else w for w in words]
    text = " ".join(new_words)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def validasi_makian(text):
    words = text.split()
    for w in words:
        if w in set_kasar:
            return True
    return False

# ==========================================
# 3. PROSES PENGGABUNGAN DATA
# ==========================================
all_texts = []
all_labels = []

# --- DATASET 1: data.csv ---
try:
    df1 = pd.read_csv('data.csv', encoding='latin-1')
    def label_ahli(row):
        text = str(row['Tweet']).lower()
        # Cek Keyword Prioritas
        for k in keywords['Pelecehan Seksual']:
            if k in text: return 'Pelecehan Seksual'
        for k in keywords['Ekonomi']:
            if k in text: return 'Ekonomi'

        # Cek Kolom Ahli
        if row['HS_Religion'] == 1: return 'Agama'
        if row['HS_Race'] == 1: return 'Ras'
        if row['HS_Physical'] == 1: return 'Fisik'
        if row['HS_Gender'] == 1: return 'Gender'
        if row['HS_Other'] == 1 or row['HS_Strong'] == 1: return 'Hate Speech / Ancaman'

        # VALIDASI MAKIAN
        if row['Abusive'] == 1:
            for k in keywords['Fisik']:
                if k in text: return 'Fisik'
            return 'Makian'

        return 'Non-Bullying'

    df1['Label'] = df1.apply(label_ahli, axis=1)
    all_texts.extend(df1['Tweet'].tolist())
    all_labels.extend(df1['Label'].tolist())
    print("✅ data.csv diproses.")
except: pass

# --- DATASET 2: Combined Dataset ---
try:
    df2 = pd.read_csv('combined_dataset.csv')

    def label_combined(row):
        text = str(row['clean_text']).lower()
        label_asal = str(row['Label']).lower()

        # PERCAYA LABEL NON-BULLYING
        if label_asal in ['non-bullying', 'positive', 'positif']:
            return 'Non-Bullying'

        # JIKA LABELNYA BULLYING, CARI JENISNYA
        for kat, k_list in keywords.items():
            for k in k_list:
                if k in text: return kat

        # Cek Keyword Dasar
        if 'kafir' in text: return 'Agama'
        if 'cina' in text or 'pribumi' in text: return 'Ras'

        # --- LOGIKA BARU UNTUK MAKIAN -
        if validasi_makian(text):
            return 'Makian'

        return 'Non-Bullying'

    df2['Label'] = df2.apply(label_combined, axis=1)
    all_texts.extend(df2['clean_text'].tolist())
    all_labels.extend(df2['Label'].tolist())
    print("✅ combined_dataset diproses.")
except: pass

# ==========================================
# 4. FINAL CLEANING
# ==========================================
df_final = pd.DataFrame({'text': all_texts, 'Label_Final': all_labels})

print("⏳ Cleaning text...")
df_final['text_clean'] = df_final['text'].apply(cleaning)

df_final = df_final.drop_duplicates(subset=['text_clean'])
df_final = df_final.dropna()

print(f"\n📊 TOTAL DATA: {len(df_final)}")
print(df_final['Label_Final'].value_counts())

df_final[['text', 'Label_Final']].to_csv('DATASET_UTAMA.csv', index=False)

