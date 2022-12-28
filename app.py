# Flask API creation

import pickle

from flask import Flask, request ,jsonify
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import numpy as np

model = pickle.load(open('moviereviews.pkl', 'rb'))
premodel=pickle.load(open('premodel.pkl','rb'))
cv=pickle.load(open('counter.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def start():
    return "hello world"


@app.route('/predict',methods=['POST'])
def predict():
    review = request.form.get('review')
    # res = model.predict([url])[0]
    f1 = premodel.clean(review)
    f2 = premodel.is_special(f1)
    f3 = premodel.to_lower(f2)
    f4 = premodel.rem_stopwords(f3)
    f5 = premodel.stem_txt(f4)


    bow,words = [],word_tokenize(f5)
    for word in words:
        bow.append(words.count(word))    
        word_dict = cv.vocabulary_

    inp = []
    for i in word_dict:
        inp.append(f5.count(i[0]))

    y_pred=model.predict(np.array(inp).reshape(1,12500))

    return jsonify({'result':str(y_pred[0])})


if __name__ == '__main__':
    app.run(debug=True)