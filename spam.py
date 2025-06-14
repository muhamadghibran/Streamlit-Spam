import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import re
import string

st.set_page_config(
    page_title="Aplikasi Deteksi Spam",
    page_icon="📧",
    layout="wide"
)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def load_or_train_model():
    model_path = 'model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, True
    else:
        try:
            df = pd.read_csv('SMSSpamCollection.txt', sep='\t', header=None, names=['label', 'message'])
            df['message'] = df['message'].apply(preprocess_text)

            X_train, X_test, y_train, y_test = train_test_split(
                df['message'], df['label'], test_size=0.2, random_state=42
            )

            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            model = MultinomialNB()
            model.fit(X_train_tfidf, y_train)

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)

            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            return model, vectorizer, False, accuracy
            
        except FileNotFoundError:
            st.error("Dataset 'SMSSpamCollection.txt' tidak ditemukan. Pastikan file dataset ada di folder yang sama.")
            return None, None, False

def predict_spam(message, model, vectorizer):
    processed_message = preprocess_text(message)

    message_tfidf = vectorizer.transform([processed_message])

    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]
    
    return prediction, probability

def main():
    st.title("📧 Aplikasi Deteksi Pesan Spam")
    st.markdown("---")

    st.sidebar.title("Tentang Aplikasi")
    st.sidebar.info(
        "Aplikasi mendeteksi pesan spam. Model dilatih menggunakan dataset "
        "SMS Spam Collection untuk mengklasifikasi pesan sebagai "
        "SPAM atau HAM (bukan spam)."
    )

    with st.spinner("Memuat model..."):
        result = load_or_train_model()
        
        if result is None or result[0] is None:
            st.error("Gagal memuat atau melatih model.")
            return
        
        if len(result) == 3:
            model, vectorizer, is_loaded = result
            if is_loaded:
                st.success("✅ Model berhasil dimuat dari file yang sudah tersimpan.")
            else:
                st.success("✅ Model berhasil dilatih dan disimpan.")
        else:
            model, vectorizer, is_loaded, accuracy = result
            if not is_loaded:
                st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.4f}")

    st.header("🔍 Deteksi Pesan Spam")

    input_method = st.radio(
        "Pilih metode input:",
        ["Input Teks", "Area Teks"]
    )
    
    if input_method == "Input Teks":
        message = st.text_input(
            "Masukkan pesan yang ingin dianalisis:",
            placeholder="Ketik pesan di sini..."
        )
    else:
        message = st.text_area(
            "Masukkan pesan yang ingin dianalisis:",
            height=100,
            placeholder="Ketik atau tempel pesan panjang di sini..."
        )

    if st.button("🔍 Analisis Pesan", type="primary"):
        if message.strip():
            with st.spinner("Menganalisis pesan..."):
                prediction, probability = predict_spam(message, model, vectorizer)

                st.markdown("---")
                st.subheader("📊 Hasil Analisis")

                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 'spam':
                        st.error("🚨 **TERDETEKSI SPAM**")
                        spam_prob = max(probability) * 100
                        st.metric("Probabilitas Spam", f"{spam_prob:.2f}%")
                    else:
                        st.success("✅ **HAM (Bukan Spam)**")
                        ham_prob = max(probability) * 100
                        st.metric("Probabilitas Ham", f"{ham_prob:.2f}%")
                
                with col2:
                    st.subheader("Detail Probabilitas")
                    prob_df = pd.DataFrame({
                        'Kelas': ['Ham', 'Spam'],
                        'Probabilitas': [probability[0]*100, probability[1]*100]
                    })
                    st.bar_chart(prob_df.set_index('Kelas'))

                st.subheader("📝 Pesan yang Dianalisis")
                st.text_area("", value=message, height=100, disabled=True)
                
        else:
            st.warning("⚠️ Silakan masukkan pesan terlebih dahulu.")

    st.markdown("---")
    st.subheader("💡 Contoh Pesan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Contoh Pesan Spam:**")
        spam_examples = [
            "MENDESAK! Anda memenangkan hadiah $1000! Klik di sini untuk mengklaim sekarang!",
            "GRATIS! Telepon sekarang untuk upgrade ponsel gratis!",
            "Menangkan hadiah uang tunai! Kirim SMS KE 12345 sekarang!",
            "Selamat! Anda terpilih untuk penawaran khusus!"
        ]
        
        for i, example in enumerate(spam_examples, 1):
            if st.button(f"Tes Spam {i}", key=f"spam_{i}"):
                st.session_state['test_message'] = example
    
    with col2:
        st.markdown("**Contoh Pesan Ham:**")
        ham_examples = [
            "Hai, apa kabar hari ini?",
            "Jangan lupa rapat besok jam 2 siang ya",
            "Terima kasih sudah membantu proyeknya",
            "Bisa tolong belikan susu dalam perjalanan pulang?"
        ]
        
        for i, example in enumerate(ham_examples, 1):
            if st.button(f"Tes Ham {i}", key=f"ham_{i}"):
                st.session_state['test_message'] = example

    if 'test_message' in st.session_state:
        st.experimental_rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p><small>Model: Naive Bayes | Dataset: SMS Spam Collection</small></p>
            <p><small>Tugas Machine Learning - Teknik Informatika</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()