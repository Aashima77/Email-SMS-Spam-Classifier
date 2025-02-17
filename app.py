import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt') 

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Remove special characters
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load trained model and vectorizer
try:
    model = joblib.load("model.joblib")
    tfidf = joblib.load("vectorizer.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        try:
            # Preprocess the input message
            transformed_sms = transform_text(input_sms)

            # Vectorize the input message
            vector_input = tfidf.transform([transformed_sms])

            # Predict the result
            result = model.predict(vector_input)[0]

            # Display the result
            if result == 1:
                st.header("🚨 Spam")
            else:
                st.header("✅ Not Spam")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
