# import libraries
import pandas as pd
import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Load your data frame samples
fake_path = "/content/Fake.csv"  # Adjust the file path based on where you uploaded the file
fake_sample = pd.read_csv(fake_path, encoding="latin1", on_bad_lines='skip')

true_path = "/content/True.csv"  # Adjust the file path based on where you uploaded the file
true_sample = pd.read_csv(true_path, encoding="latin1", on_bad_lines='skip')

# Load your saved models using pickle
svm_model_path = "/content/svm_model.pkl"  # Adjust the path based on where you uploaded the file
with open(svm_model_path, "rb") as f:
    svm_model = pickle.load(f)

# Load the vectorizer
vectorizer_path = "/content/tfid_algo.sav"  # Adjust the path based on where you uploaded the file
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)
# Create a function to clean text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\\W', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Prediction function
def news_prediction(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    new_x_test = new_def_test['text']
    new_tfidf_test = vectorizer.transform(new_x_test)
    pred_dt = svm_model.predict(new_tfidf_test)

    if pred_dt[0] == 0:
        return "This is Fake News!"
    else:
        return "The News seems to be True!"

def main():
    # Write the app title and introduction
    st.title("Fake News Prediction System")
    st.write("Context: ... (your description)")

    # User input area
    user_text = st.text_area('Text to Analyze', '''(paste news text here)''', height=350)
    
    # Button to trigger analysis
    if st.button("Article Analysis Result"):
        news_pred = news_prediction(user_text)
        if news_pred == "This is Fake News!":
            st.error(news_pred, icon="🚨")
        else:
            st.success(news_pred)
            st.balloons()

    # Sample articles section
    st.write("## Sample Articles to Try:")
    st.write("#### Fake News Article")
    st.write("Click the box below and copy/paste.")
    st.dataframe(fake_sample['text'].sample(1), hide_index=True)

    st.write("#### Real News Article")
    st.write("Click the box below and copy/paste.")
    st.dataframe(true_sample['text'].sample(1), hide_index=True)

if __name__ == "__main__":
    main()