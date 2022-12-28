# Flask API creation

import pickle

from flask import Flask, request ,jsonify
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import numpy as np

model = pickle.load(open('moviereviews.pkl', 'rb'))
premodel=pickle.load(open('premodel.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def start():
    return "hello world"


@app.route('/predict',methods=['POST'])
def predict():
    review = request.form.get('review')
    # res = model.predict([url])[0]
    f1 = premodel.clean(rev)
    f2 = premodel.is_special(f1)
    f3 = premodel.to_lower(f2)
    f4 = premodel.rem_stopwords(f3)
    f5 = premodel.stem_txt(f4)

    cv = CountVectorizer()

    bow,words = [],word_tokenize(f5)
    for word in words:
        bow.append(words.count(word))    
        word_dict = cv.vocabulary_
    y_pred=np.array(review).reshape(1,12500)[0]
    
    return jsonify({'result':str(res)})


if __name__ == '__main__':
    app.run(debug=True)