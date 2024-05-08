import pandas as pd
import pymorphy2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self):
        print('downloading libraries...')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = [self.morph.parse(token)[0].normal_form for token in tokens if token.isalnum()]
        filtered_tokens = [token for token in lemmatized_tokens if token not in self.stop_words]
        return ' '.join(filtered_tokens)

    def load_data(self):
        print('reading from excel...')
        dfn = pd.read_excel('negative_excel.xlsx', sheet_name='negativeansi')
        dfp = pd.read_excel('positive_excel.xlsx', sheet_name='positiveansi')
        print('preprocessing text...')
        neg_x_arr = list(dfn['tweet'].apply(str).apply(self.preprocess_text))
        pos_x_arr = list(dfp['tweet'].apply(str).apply(self.preprocess_text))
        neg_y_arr = ['negative' for _ in range(len(neg_x_arr))]
        pos_y_arr = ['positive' for _ in range(len(pos_x_arr))]
        x_arr = neg_x_arr + pos_x_arr
        y_arr = neg_y_arr + pos_y_arr
        return train_test_split(x_arr, y_arr, test_size=0.2, random_state=42)
