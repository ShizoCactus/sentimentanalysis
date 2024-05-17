from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


class LogisticRegressionModel:
    def __init__(self, x_train, x_test, y_train, y_test=None, n=1, vectortype='count'):
        # Инициализация модели логистической регрессии
        self.X_train = x_train  # Обучающая выборка
        self.X_test = x_test  # Тестовая выборка
        self.y_train = y_train  # Метки обучающей выборки
        self.y_test = y_test  # Метки тестовой выборки
        self.x_train_vectorized = None  # Векторизованные обучающие данные
        self.x_test_vectorized = None  # Векторизованные тестовые данные
        self.n = n  # Размер n-грамм
        self.vectortype = vectortype  # Тип векторизации ('count' или 'tfidf')
        self.y_pred_ngrams = None  # Предсказанные метки

    def vectorize(self):
        # Векторизация данных с использованием CountVectorizer
        print('Векторизация ' + str(self.n) + '-грамм...')
        vectorizer = CountVectorizer(ngram_range=(self.n, self.n))  # Инициализация CountVectorizer для n-грамм
        self.x_train_vectorized = vectorizer.fit_transform(self.X_train)  # Векторизация обучающих данных
        self.x_test_vectorized = vectorizer.transform(self.X_test)  # Векторизация тестовых данных

    def tfidf_vectorize(self):
        # Векторизация данных с использованием TfidfVectorizer
        print('TF-IDF векторизация ' + str(self.n) + '-грамм...')
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(self.n, self.n))  # Инициализация TfidfVectorizer для n-грамм
        self.x_train_vectorized = tfidf_vectorizer.fit_transform(self.X_train)  # Векторизация обучающих данных
        self.x_test_vectorized = tfidf_vectorizer.transform(self.X_test)  # Векторизация тестовых данных

    def process(self):
        # Процесс векторизации и обучения модели логистической регрессии
        if self.vectortype == 'tfidf':
            self.tfidf_vectorize()  # Векторизация с использованием TF-IDF
        else:
            self.vectorize()  # Векторизация с использованием CountVectorizer
        print('Обработка...')
        lr = LogisticRegression(max_iter=1000)  # Инициализация модели логистической регрессии
        lr.fit(self.x_train_vectorized, self.y_train)  # Обучение модели
        self.y_pred_ngrams = lr.predict(self.x_test_vectorized)  # Предсказание меток для тестовой выборки

    def print_results(self):
        # Вывод результатов предсказания для каждого тестового примера
        for i in range(len(self.y_pred_ngrams)):
            print(self.X_test[i] + ': ' + self.y_pred_ngrams[i])

    def report(self):
        # Вывод отчета по классификации
        if self.vectortype == 'tfidf':
            print("Отчет о классификации для TF-IDF " + str(self.n) + '-грамм:')
        else:
            print("Отчет о классификации для " + str(self.n) + '-грамм:')
        print(classification_report(self.y_test, self.y_pred_ngrams))  # Вывод метрик классификации
