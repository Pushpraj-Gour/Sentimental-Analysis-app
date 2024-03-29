import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st

nltk.download("stopwords")
clr = pickle.load(open("model.sav","rb"))
vector = pickle.load(open("vector.sav","rb"))
# print("Hello")

port_stem = PorterStemmer()
# Load the function from the pickle file

def stemming(tweet):
  new_tweet = re.sub("[^a-zA-Z]"," ",tweet)
  new_tweet = new_tweet.lower()
  new_tweet = new_tweet.split()
  new_tweet = [port_stem.stem(word) for word in new_tweet if not word in stopwords.words("english")]
  new_tweet = " ".join(new_tweet)
  return new_tweet

def predictor(text):
    new_text= stemming(text)
    new_text = vector.transform([text])
    result = clr.predict(new_text)
    if result == 4:
        return f"'{text}'--> it's a Positive Review"
    else:
        return f"'{text}'--> It's a negative Review"

def main():

    # Giving the title
    st.title("Sentiment Analysis")


    # Getting the input data from user

    user_review = st.text_input("Your Review")

    #Code for prediction
    response = ''


    if st.button("Sentiment"):
        response = predictor(user_review)

    st.success(response)


if __name__=="__main__":
    main()
