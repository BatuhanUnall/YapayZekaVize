import tkinter as tk
from tkinter import messagebox, scrolledtext
import csv
import numpy as np
import os
import re
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import nltk
import urllib.parse

# 🚨 GEREKLİ KÜTÜPHANELERİN KONTROLÜ VE IMPORT 🚨
try:
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
    import yt_dlp
except ImportError:
    # Eğer kütüphaneler yoksa uyarı verip kapatır
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Kütüphane Eksik",
                         "Lütfen terminale şu komutu yapıştırıp enter'a basın:\n\npip install youtube-comment-downloader yt-dlp")
    exit()

# ----------------------------------------------------------------------
# 0. GLOBAL DEĞİŞKENLER
# ----------------------------------------------------------------------
df_cekilen_yorumlar = None
MAX_YORUM_SAYISI = 40  # Ödev için yeterli sayı

# ----------------------------------------------------------------------
# 1. TEMEL NLP FONKSİYONLARI
# ----------------------------------------------------------------------
try:
    turkish_stopwords = stopwords.words('turkish')
except LookupError:
    nltk.download('stopwords', quiet=True)
    turkish_stopwords = stopwords.words('turkish')


def metin_temizle_ve_tokenlestir(metin):
    metin = str(metin).lower()
    metin = re.sub(r'[^a-zıüşöçğ\s]', '', metin)
    tokenler = metin.split()
    tokenler = [kelime for kelime in tokenler if kelime not in turkish_stopwords and len(kelime) > 1]
    return tokenler


def yorum_vektoru_olustur(token_listesi, model, vector_size):
    vektorler = [model.wv[kelime] for kelime in token_listesi if kelime in model.wv]
    if len(vektorler) == 0:
        return np.zeros(vector_size)
    else:
        return np.mean(vektorler, axis=0)


# ----------------------------------------------------------------------
# 2. MODEL YÜKLEME
# ----------------------------------------------------------------------
MODEL_DIZINI = 'kayitli_modeller'
W2V_MODEL_YOLU = os.path.join(MODEL_DIZINI, "word2vec_model.model")
MLP_MODEL_YOLU = os.path.join(MODEL_DIZINI, "mlp_model_en_iyi.keras")

MODEL_YUKLU = False
try:
    w2v_model = Word2Vec.load(W2V_MODEL_YOLU)
    mlp_model = load_model(MLP_MODEL_YOLU)
    VEKTOR_BOYUTU = w2v_model.vector_size
    MODEL_YUKLU = True
except Exception as e:
    pass  # Hata arayüz açılınca gösterilecek


# ----------------------------------------------------------------------
# 3. VERİ ÇEKME FONKSİYONLARI (PDF GEREKSİNİMLERİ)
# ----------------------------------------------------------------------

def get_video_id(url):
    """URL'den Video ID'sini ayrıştırır."""
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return urllib.parse.parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed_url.path[1:]
    return url


def video_bilgilerini_getir(url):
    """
    Videonun yayınlanma tarihi, yayınlayan kişi ve beğeni sayısını çeker.
    (PDF Madde 7)
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Tarihi düzelt (YYYYMMDD -> YYYY-MM-DD)
            upload_date = info.get('upload_date', 'Bilinmiyor')
            if len(upload_date) == 8:
                formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
            else:
                formatted_date = upload_date

            return {
                "Video_Baslik": info.get('title', 'Bilinmiyor'),
                "Yayinlayan_Kullanici": info.get('uploader', 'Bilinmiyor'),
                "Yayinlanma_Tarihi": formatted_date,
                "Video_Begeni": info.get('like_count', 0)
            }
    except Exception as e:
        return f"Hata: Video bilgileri çekilemedi. {e}"


def yorumlari_cek(video_url, video_meta):
    """
    Yorumları çeker ve video bilgileriyle birleştirir.
    (PDF Madde 9-13)
    """
    video_id = get_video_id(video_url)
    if not video_id: return "Hata: Geçersiz URL."

    downloader = YoutubeCommentDownloader()
    try:
        comments = downloader.get_comments_from_url(
            f'https://www.youtube.com/watch?v={video_id}',
            sort_by=SORT_BY_POPULAR
        )

        veri_listesi = []
        for i, comment in enumerate(comments):
            # Her yorum satırına video bilgisini de ekliyoruz (İlişkisel yapı)
            veri = {
                # Video Bilgileri (Üst Veri)
                "Video_Sahibi": video_meta["Yayinlayan_Kullanici"],
                "Video_Tarihi": video_meta["Yayinlanma_Tarihi"],
                "Video_Begeni": video_meta["Video_Begeni"],

                # Yorum Bilgileri
                "Yorum_Yazan": comment.get("author", "Anonim"),
                "Yorum_Metni": comment.get("text", ""),
                "Yorum_Tarihi": comment.get("time_parsed", "Bilinmiyor"),  # Göreceli zaman (örn: 2 hafta önce)
                "Yorum_Begeni": comment.get("votes", 0),  # Like Sayısı
            }
            veri_listesi.append(veri)

            if len(veri_listesi) >= MAX_YORUM_SAYISI:
                break

        if not veri_listesi: return "Hata: Yorum bulunamadı."
        return veri_listesi

    except Exception as e:
        return f"Yorum çekme hatası: {e}"


# --- TAHMİN VE ANALİZ ---
def tahmin_et(dataframe, ekran):
    if not MODEL_YUKLU:
        messagebox.showerror("Hata", "Modeller yüklü değil. Lütfen main.py'yi çalıştırın.")
        return

    ekran.insert(tk.END, "\n" + "=" * 40 + "\n")
    ekran.insert(tk.END, "🤖 DUYGU ANALİZİ SONUÇLARI\n")
    ekran.insert(tk.END, "=" * 40 + "\n")

    # Ön işleme
    dataframe['tokens'] = dataframe['Yorum_Metni'].apply(metin_temizle_ve_tokenlestir)
    dataframe['vector'] = dataframe['tokens'].apply(
        lambda x: yorum_vektoru_olustur(x, w2v_model, VEKTOR_BOYUTU)
    )

    # Tahmin
    X_pred = np.stack(dataframe['vector'].values)
    preds = mlp_model.predict(X_pred, verbose=0)
    classes = np.argmax(preds, axis=1)

    labels = {0: "NEGATİF 😡", 1: "POZİTİF 😊", 2: "NÖTR 😐"}

    for i, row in dataframe.iterrows():
        sonuc = labels.get(classes[i], "Bilinmiyor")
        ekran.insert(tk.END, f"👤 {row['Yorum_Yazan']}:\n")
        ekran.insert(tk.END, f"📝 \"{row['Yorum_Metni']}\"\n")  # Tam Metin
        ekran.insert(tk.END, f"📊 Tahmin: {sonuc}\n")
        ekran.insert(tk.END, "-" * 40 + "\n")

    messagebox.showinfo("Bitti", "Analiz tamamlandı.")


# --- ARAYÜZ OLUŞTURMA (GUI) ---
def create_gui():
    global df_cekilen_yorumlar

    root = tk.Tk()
    root.title("Yapay Zeka Ödevi - Video Analiz Arayüzü")
    root.geometry("800x750")

    if not MODEL_YUKLU:
        lbl = tk.Label(root, text="⚠️ Modeller Yüklenemedi! main.py çalıştırılmalı.", fg="red")
        lbl.pack()

    # Başlık
    tk.Label(root, text="YouTube Duygu Analizi ve Veri Çekme", font=("Helvetica", 14, "bold")).pack(pady=10)

    # URL Giriş
    frame_url = tk.Frame(root)
    frame_url.pack(pady=5)
    tk.Label(frame_url, text="Video URL:").pack(side=tk.LEFT, padx=5)
    entry_url = tk.Entry(frame_url, width=50)
    entry_url.pack(side=tk.LEFT, padx=5)

    # Bilgi Ekranı
    # Bu kısım YORUMLARIN ALT SATIRA İNMESİNİ sağlar:
    # 'wrap=tk.WORD' ile kelime boşluklarından satır atlaması yapılır.
    text_area = scrolledtext.ScrolledText(root, width=90, height=30, font=("Consolas", 9), wrap=tk.WORD)
    text_area.pack(pady=10)

    # Buton Fonksiyonu
    def verileri_cek_btn():
        global df_cekilen_yorumlar
        url = entry_url.get().strip()
        text_area.delete(1.0, tk.END)

        if not url:
            messagebox.showwarning("Uyarı", "Lütfen bir URL girin.")
            return

        text_area.insert(tk.END, "⏳ Video bilgileri ve yorumlar çekiliyor, lütfen bekleyin...\n")
        root.update()

        # 1. Video Bilgisi
        video_meta = video_bilgilerini_getir(url)
        if isinstance(video_meta, str):
            text_area.insert(tk.END, video_meta + "\n")
            return

        # 2. Yorumlar
        veri = yorumlari_cek(url, video_meta)
        if isinstance(veri, str):
            text_area.insert(tk.END, veri + "\n")
            return

        df_cekilen_yorumlar = pd.DataFrame(veri)

        # CSV Kaydet
        csv_name = "cekilen_yorumlar.csv"
        df_cekilen_yorumlar.to_csv(csv_name, sep=';', index=False, encoding='utf-8-sig')

        # Ekrana Yazdır
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, f"✅ VİDEO BİLGİLERİ (Kaydedildi: {csv_name})\n")
        text_area.insert(tk.END, f"Başlık: {video_meta['Video_Baslik']}\n")
        text_area.insert(tk.END, f"Yayınlayan: {video_meta['Yayinlayan_Kullanici']}\n")
        text_area.insert(tk.END, f"Tarih: {video_meta['Yayinlanma_Tarihi']}\n")
        text_area.insert(tk.END, f"Video Beğeni: {video_meta['Video_Begeni']}\n")
        text_area.insert(tk.END, "=" * 60 + "\n\n")

        text_area.insert(tk.END, f"✅ ÇEKİLEN YORUMLAR (İlk {len(df_cekilen_yorumlar)} adet)\n")
        text_area.insert(tk.END, "-" * 60 + "\n")
        for i, row in df_cekilen_yorumlar.iterrows():
            text_area.insert(tk.END, f"[{i + 1}] {row['Yorum_Yazan']} ({row['Yorum_Tarihi']}):\n")
            text_area.insert(tk.END, f"   \"{row['Yorum_Metni']}\"\n")  # Tam Metin
            text_area.insert(tk.END, f"   (Beğeni: {row['Yorum_Begeni']})\n")
            text_area.insert(tk.END, "-" * 60 + "\n")

    def analiz_et_btn():
        global df_cekilen_yorumlar
        if df_cekilen_yorumlar is None:
            messagebox.showwarning("Uyarı", "Önce verileri çekmelisiniz.")
            return
        tahmin_et(df_cekilen_yorumlar, text_area)

    # Butonlar
    frame_btn = tk.Frame(root)
    frame_btn.pack(pady=10)

    btn_cek = tk.Button(frame_btn, text="1. Verileri Çek ve Kaydet", command=verileri_cek_btn, bg="#dddddd")
    btn_cek.pack(side=tk.LEFT, padx=10)

    btn_analiz = tk.Button(frame_btn, text="2. Duygu Analizi Yap", command=analiz_et_btn, bg="#aaffaa")
    btn_analiz.pack(side=tk.LEFT, padx=10)

    root.mainloop()


if __name__ == "__main__":
    create_gui()