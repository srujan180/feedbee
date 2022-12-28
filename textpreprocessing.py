import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

class Preprocessing:
    @staticmethod
    def clean(text):
        cleaned = re.compile(r'<.*?>')
        return re.sub(cleaned,'',text)
    

    @staticmethod
    def is_special(text):
        rem = ''
        for i in text:
            if i.isalnum():
                rem = rem + i
            else:
                rem = rem + ' '
        return rem

    @staticmethod
    def to_lower(text):
        return text.lower()

    @staticmethod
    def rem_stopwords(text):
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        return [w for w in words if w not in stop_words]

    @staticmethod
    def stem_txt(text):
        ss = SnowballStemmer('english')
        return " ".join([ss.stem(w) for w in text])
    
