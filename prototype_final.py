import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

def main():
    st.title("Prototype Prediksi Kode Klasifikasi Arsip")
    
    # Sidebar
    # st.sidebar.title("Konfigurasi Model")
    # classifier_name = st.sidebar.selectbox("Pilih Model Klasifikasi", ("Model 1", "Model 2", "Model 3"))

    # Load pre-trained model
    classifier = joblib.load("naive_bayes_model_final.pkl")
    tfidf = joblib.load("naive_bayes_tfidf_final.pkl")
    
    # Input teks
    text_input = st.text_input("Masukkan hal surat", "")

    # Model prediction
    if st.button("Prediksi"):
        if text_input != "":
            #mengubah data ke huruf kecil semua dan menyimpan ke kolom uraian_informasi_arsip_lower
            text_input = text_input.lower()
            # Hapus angka dan tanda baca menggunakan regular expressions, dan menambahkan kolom uraian_informasi_arsip_lower_clean
            text_input = re.sub(r'[^\w\s]', ' ', re.sub(r'\d+', '', text_input))
            # Hapus stopword, untuk menghapus kata yang tidak bermakna, kata sambung dalam bahasa indonesia
            stopwords_ind = set(stopwords.words('indonesian'))
            #untuk menghapus stopwords/ kata yang dianggap tidak penting serta menyimpan hasilnya ke dalam variabel text_input
            text_input_list = [x for x in text_input.split() if x not in list(stopwords_ind)]
            # Konversi data preprocessed menjadi string, menggabungkan setiap kata / token
            text_input_preprocessed = ' '.join(text_input_list)
            teks = tfidf.transform([text_input_preprocessed])
            prediction = classifier.predict(teks)
            st.write("Prediksi:")
            st.write(prediction[0])
        else:
            st.write("Mohon masukkan teks terlebih dahulu.")

if __name__ == "__main__":
    main()