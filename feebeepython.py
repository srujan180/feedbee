import numpy as np 
import pandas as pd 
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle


# # ntk downloader
# nltk.download('punkt')
# nltk.download('stopwords')
data = pd.read_csv('dataset/IMDB-Dataset.csv')


data.sentiment.replace('positive',1,inplace=True)
data.sentiment.replace('negative',0,inplace=True)


def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

data.review = data.review.apply(clean)
data.review[0]




def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

data.review = data.review.apply(is_special)
data.review[0]




def to_lower(text):
    return text.lower()

data.review = data.review.apply(to_lower)
data.review[0]


from nltk.corpus import stopwords
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

data.review = data.review.apply(rem_stopwords)

    