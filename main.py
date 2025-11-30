import pandas as pd
import nltk
import numpy as np
import re
import os
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# 1. AYARLAR VE DOSYA LİSTESİ (✅ TÜM VERİ KAYNAKLARINIZ BURAYA EKLENMİŞTİR)
# ----------------------------------------------------------------------
# Proje klasöründeki tüm eğitim dosyalarının adları.
DOSYA_LISTESI = [
    'e-ticaret_urun_yorumlari.csv',  # 1. Orijinal e-ticaret verisi
    'kendi_kayitlarim.csv',  # 2. Kendi etiketlediğim YouTube verisi
    'train.csv',  # 3. Geniş sosyal medya verisi

]
VEKTOR_BOYUTU = 100
EGITIM_EPOCH = 12

# ----------------------------------------------------------------------
# 2. NLTK KAYNAKLARINI İNDİRME
# ----------------------------------------------------------------------
try:
    print("NLTK kaynakları kontrol ediliyor (stopwords, punkt)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"❌ NLTK hatası: {e}")
    exit()

# ----------------------------------------------------------------------
# 3. CSV OKUMA VE ETİKET STANDARTLAŞTIRMA MOTORU
# ----------------------------------------------------------------------
print(f"\nToplam {len(DOSYA_LISTESI)} adet veri dosyası okunacak...")

df = pd.DataFrame()
toplam_okunan_dosya = 0

for dosya_adi in DOSYA_LISTESI:
    if not os.path.exists(dosya_adi):
        print(f"⚠️ Dosya bulunamadı, atlanıyor: {dosya_adi}")
        continue

    temp_df = None
    # Farklı ayırıcı ve kodlama denemeleri
    okuma_parametreleri = [
        {'sep': ';', 'encoding': 'iso-8859-9'},
        {'sep': ',', 'encoding': 'utf-8'},
        {'sep': ',', 'encoding': 'iso-8859-9'},
        {'sep': ';', 'encoding': 'utf-8'}
    ]

    for params in okuma_parametreleri:
        try:
            temp_df = pd.read_csv(dosya_adi, **params)
            # UTF-8 BOM karakterini temizle
            temp_df.columns = temp_df.columns.str.replace('ï»¿', '').str.strip()
            break
        except:
            continue

    if temp_df is not None:
        print(f"📄 '{dosya_adi}' okundu. Sütunlar: {list(temp_df.columns)}")

        # --- SÜTUN EŞLEŞTİRME ---
        # Metin sütunu için olası isimler (Yorum, text, Yorum vb.)
        olasi_metin = ['Metin', 'text', 'yorum', 'Yorum', 'comment', 'Görüş', 'content', 'Text']
        # Etiket sütunu için olası isimler (Durum, label, Duygu vb.)
        olasi_etiket = ['Durum', 'label', 'sentiment', 'target', 'class', 'duygu', 'Label', 'Sentiment', 'Duygu']

        # Olası sütun adlarını bul ve standart isimlerle eşle
        bulunan_metin = next((col for col in temp_df.columns if col in olasi_metin), None)
        bulunan_etiket = next((col for col in temp_df.columns if col in olasi_etiket), None)

        if bulunan_metin and bulunan_etiket:
            temp_df = temp_df.rename(columns={bulunan_metin: 'Metin', bulunan_etiket: 'Durum'})
            df = pd.concat([df, temp_df[['Metin', 'Durum']]], ignore_index=True)
            print(f"   ✅ Eklendi: {len(temp_df)} satır.")
            toplam_okunan_dosya += 1
        else:
            print(f"   ❌ UYARI: '{dosya_adi}' içinde uygun Metin/Etiket sütunu bulunamadı.")
    else:
        print(f"   ❌ HATA: '{dosya_adi}' okunamadı (Format hatası).")

if toplam_okunan_dosya == 0:
    print("\n❌ HİÇBİR VERİ OKUNAMADI. Program durduruluyor.")
    exit()

# Veri temizliği (Boş satırları sil)
df = df.dropna(subset=['Metin', 'Durum'])

# --- KRİTİK KISIM: ETİKET STANDARTLAŞTIRMA (Tüm formatları 1/0/2'ye çevirir) ---
print("\nEtiketler (Durum sütunu) standartlaştırılıyor (Text, 0, 1, 2'den -> 1/0/2'ye)...")

# Kesin Eşleşme Sözlüğü
etiket_esleme_sozlugu = {
    # Negatif (0)
    '0': 0, 'olumsuz': 0, 'negatif': 0, 'negative': 0, 'negative': 0, 'neg': 0,
    # Pozitif (1)
    '1': 1, 'olumlu': 1, 'pozitif': 1, 'positive': 1, 'positive': 1, 'pos': 1,
    # Nötr/Tarafsız (2)
    '2': 2, 'nötr': 2, 'tarafsız': 2, 'neutral': 2, 'notr': 2, 'neu': 2
}


def etiket_duzelt_guclu(deger):
    # Değeri metne çevirip küçük harfe dönüştür ve boşlukları temizle
    str_val = str(deger).strip().lower()

    if str_val in etiket_esleme_sozlugu:
        return etiket_esleme_sozlugu[str_val]

    # Eğer etiket hiçbir şeye benzemiyorsa, varsayılan olarak Nötr (2) yap.
    return 2


df['Durum'] = df['Durum'].apply(etiket_duzelt_guclu)

# Etiketleri tam sayı (integer) formatına çevir
df['Durum'] = df['Durum'].astype(int)

YORUM_SUTUNU_ADI = 'Metin'
ETIKET_SUTUNU_ADI = 'Durum'
print(f"\n✅ BİRLEŞTİRME VE STANDARTLAŞTIRMA TAMAMLANDI. Toplam Eğitim Verisi: {len(df)} yorum.")
# ----------------------------------------------------------------------
# ... (KODUN DEVAMI AŞAĞIDADIR)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 4. ÖN İŞLEME VE WORD2VEC EĞİTİMİ
# ----------------------------------------------------------------------
turkish_stopwords = stopwords.words('turkish')


def metin_temizle_ve_tokenlestir(metin):
    metin = str(metin).lower()
    # Türkçe harfleri ve boşlukları koru
    metin = re.sub(r'[^a-zıüşöçğ\s]', '', metin)
    tokenler = metin.split()
    tokenler = [kelime for kelime in tokenler if kelime not in turkish_stopwords and len(kelime) > 1]
    return tokenler


print("\nYorumlar temizleniyor ve tokenleştiriliyor...")
df['temiz_tokenler'] = df[YORUM_SUTUNU_ADI].apply(metin_temizle_ve_tokenlestir)
word2vec_corpus = df['temiz_tokenler'].tolist()

print(f"\nWord2Vec modeli {VEKTOR_BOYUTU} boyutunda eğitiliyor...")
word2vec_model = Word2Vec(
    sentences=word2vec_corpus,
    vector_size=VEKTOR_BOYUTU,
    window=5,
    min_count=2,
    sg=0,
    workers=4
)


def yorum_vektoru_olustur(token_listesi, model, vector_size):
    vektorler = [model.wv[kelime] for kelime in token_listesi if kelime in model.wv]
    if len(vektorler) == 0:
        return np.zeros(vector_size)
    else:
        return np.mean(vektorler, axis=0)


print("Cümle vektörleri oluşturuluyor...")
df['yorum_vektoru'] = df['temiz_tokenler'].apply(
    lambda tokens: yorum_vektoru_olustur(tokens, word2vec_model, VEKTOR_BOYUTU)
)

X = np.stack(df['yorum_vektoru'].values)
y = df[ETIKET_SUTUNU_ADI].values.astype(int)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-Hot Encoding
NUM_CLASSES = len(np.unique(y))
print(f"Tespit edilen sınıf sayısı: {NUM_CLASSES}")

y_train_encoded = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_encoded = to_categorical(y_test, num_classes=NUM_CLASSES)

# ----------------------------------------------------------------------
# 5. MLP MODEL EĞİTİMİ (DERİN YAPAY SİNİR AĞI)
# ----------------------------------------------------------------------
input_dim = X_train.shape[1]


def create_model_2(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


print(f"\n--- Model Eğitimi Başlıyor ({EGITIM_EPOCH} epoch) ---")
model_2 = create_model_2(input_dim, NUM_CLASSES)
model_2.fit(X_train, y_train_encoded, epochs=EGITIM_EPOCH, batch_size=32, validation_split=0.1, verbose=1)

# ----------------------------------------------------------------------
# 6. DEĞERLENDİRME VE KAYIT
# ----------------------------------------------------------------------

print(f"\n--- Test Sonuçları ---")
loss, accuracy = model_2.evaluate(X_test, y_test_encoded, verbose=0)
print(f"Test Doğruluğu: {accuracy:.4f}")

y_pred_probs = model_2.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_encoded, axis=1)

print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred, zero_division=0))
print("\nHata Matrisi:")
print(confusion_matrix(y_true, y_pred))

# KAYIT İŞLEMİ
MODEL_DIZINI = 'kayitli_modeller'
os.makedirs(MODEL_DIZINI, exist_ok=True)

word2vec_model.save(os.path.join(MODEL_DIZINI, "word2vec_model.model"))
model_2.save(os.path.join(MODEL_DIZINI, "mlp_model_en_iyi.keras"))

print("-" * 50)
print("✅ YENİLENMİŞ MODELLER BAŞARIYLA KAYDEDİLDİ.")
print("👉 Şimdi 'gui_scraper.py' dosyasını çalıştırarak YouTube yorumlarını analiz edebilirsiniz.")
print("-" * 50)

# main.py dosyasında, model_2'nin eğitimi bittikten sonra ekleyin:

print("\n--- Yapay Sinir Ağı Topolojisi (Model Özeti) ---")
model_2.summary()