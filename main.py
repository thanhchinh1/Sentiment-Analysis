import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

import streamlit as st

word_index = imdb.get_word_index()

reverse_word_index = {value:key for key, value in word_index.items()}
model = load_model('simple_rnn_imdb.keras')

def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encode_review if i > 3])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment ='Positive' if prediction [0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie review')

if st.button('Classify'):
    if user_input:
        sentiment, score = predict_sentiment(user_input)
        st.write('Sentiment: ', sentiment)
        st.write('The score: ', score)
    else:
        st.write('Please! Enter the review.')
