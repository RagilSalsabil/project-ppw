import streamlit as st
import pandas as pd
# Library
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Klasifikasi
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import string
import pickle
import joblib

st.set_page_config(
    page_title="Aplikasi Kategori Berita | Klasifikasi Berita Radar Jatim", page_icon="📺")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.image('assets/banner.png', use_column_width=True)

with col2:
    st.subheader('Sistem untuk memprediksi suatu kategori berita')

    data_new = pd.read_csv('data/radarjatim-clean.csv')

    def tokenizer(text):
        text = text.lower()
        return sent_tokenize(text)

    def create_topic_proportion_df(X_summary, k, alpha, beta):
        lda_model = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)

        proporsi_topik_dokumen = lda_model.fit_transform(X_summary)

        nama_kolom_topik = [f'Topik {i+1}' for i in range(k)]

        proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen, columns=nama_kolom_topik)

        return proporsi_topik_dokumen_df

    data_new["tokenizing"] = data_new['clean_content'].apply(tokenizer)

    # membuat kolom baru dengan nama new_abstrak untuk data baru yang dipunctuation
    data_new['clean_content'] = data_new['Content'].str.replace('[{}]'.format(string.punctuation), '').str.lower()
    # Menghilangkan angka dari kolom 'new_abstrak'
    data_new['clean_content'] = data_new['clean_content'].str.replace('\d+', '', regex=True)

    # menggabungkan kata
    data_new['final_content'] = data_new['tokenizing'].apply(lambda x: ' '.join(x))

with col3:
    # proses klasifikasi
    st.caption('Nama : MUH. RAGIL SALSABIL')
    st.caption('NIM : 200411100019')
    
    st.write("""
        ### Want to learn more?
        - Dataset (studi kasus) [radarjatim.com](https://radarjatim.id/)
        - Github Account [github.com](https://github.com/RagilSalsabil?tab=repositories)
        """)
    
    data_final_sm = pd.read_csv('data/data_final_sm.csv')

    vectorizer_summary = TfidfVectorizer()
    tfidf_text = vectorizer_summary.fit_transform(data_final_sm['summary']).toarray()
    X_summary = create_topic_proportion_df(tfidf_text, 6, 0.1, 0.2)
    y = data_final_sm["Category"]

    X_train_summary, X_test_summary, y_train_summary, y_test_summary = train_test_split(X_summary, y, test_size=0.3, random_state=42)

    # Inisialisasi model KNN
    knn_summary = KNeighborsClassifier(n_neighbors=6)  # Sesuaikan jumlah tetangga (n_neighbors) sesuai kebutuhan
    # Melatih model menggunakan data latih
    knn_summary.fit(X_train_summary, y_train_summary)
    # Membuat prediksi pada data uji
    y_pred_knn_summary = knn_summary.predict(X_test_summary)
    accuracy = accuracy_score(y_test_summary, y_pred_knn_summary)
    # print(f'Akurasi: {accuracy}')

with col4:
    st.subheader("Masukan Text")

    new_data = st.text_area("Masukkan Text Berita")


    hasil = st.button("Cek klasifikasi")

    if hasil:

        new_data_summary = tokenizer(new_data[0])
        # tfidf_Xnew_summary = vectorizer_summary.transform([new_data_summary[0]]).toarray()
        # pred_gnb_summary = gnb_summary.predict(tfidf_Xnew_summary)
        tfidf_Xnew_summary = vectorizer_summary.transform([new_data_summary[0]]).toarray()
        topik_tfidf_x = create_topic_proportion_df(tfidf_Xnew_summary, 6, 0.1, 0.2)
        
        pred_knn_summary = knn_summary.predict(topik_tfidf_x)
        pred_knn_summary[0]

        st.success(f"Prediksi Hasil Klasifikasi : {pred_knn_summary[0]}")