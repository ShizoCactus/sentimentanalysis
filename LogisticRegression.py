from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


class LogisticRegressionModel:
    def __init__(self, x_train, x_test, y_train, y_test, n, vectortype='count'):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.n = n
        self.vectortype = vectortype
        self.y_pred_ngrams = None

    def vectorize(self):
        print('vectorizing ' + str(self.n) + '-gram...')
        vectorizer = CountVectorizer(ngram_range=(self.n, self.n))
        x_train_count_ngrams = vectorizer.fit_transform(self.X_train)
        x_test_count_ngrams = vectorizer.transform(self.X_test)
        return x_train_count_ngrams, x_test_count_ngrams

    def tfidf_vectorize(self):
        print('TF-IDF vectorizing ' + str(self.n) + '-gram...')
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(self.n, self.n))
        x_train_tfidf_ngrams = tfidf_vectorizer.fit_transform(self.X_train)
        x_test_tfidf_ngrams = tfidf_vectorizer.transform(self.X_test)
        return x_train_tfidf_ngrams, x_test_tfidf_ngrams

    def process(self):
        if self.vectortype == 'tfidf':
            x_train_ngrams, x_test_ngrams = self.tfidf_vectorize()
        else:
            x_train_ngrams, x_test_ngrams = self.vectorize()
        print('processing...')
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x_train_ngrams, self.y_train)
        self.y_pred_ngrams = lr.predict(x_test_ngrams)

    def report(self):
        if self.vectortype == 'tfidf':
            print("Classification Report for TF-IDF " + str(self.n) + '-gram:')
        else:
            print("Classification Report for " + str(self.n) + '-gram:')
        print(classification_report(self.y_test, self.y_pred_ngrams))
