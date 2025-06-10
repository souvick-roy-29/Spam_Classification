import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk_path = '/tmp/nltk_data'
nltk.data.path.append(nltk_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_path)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(i) for i in text if i.isalnum() and i not in stop_words and i not in string.punctuation]
    return " ".join(y)

tfidf = pickle.load(open(r'vectorizer.pkl', 'rb'))
model = pickle.load(open(r'model.pkl', 'rb'))

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your message here")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input.toarray())[0]
    
    if result == 1:
        st.error("🚫 This is SPAM!")
    else:
        st.success("✅ This is NOT Spam!")
