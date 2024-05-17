import pandas as pd
import pymorphy2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pickle


class DataProcessor:
    def __init__(self):
        print('Загрузка необходимых библиотек...')
        nltk.download('punkt')  # Загрузка модели токенизатора punkt
        nltk.download('stopwords')  # Загрузка списка стоп-слов
        self.morph = pymorphy2.MorphAnalyzer()  # Инициализация морфологического анализатора
        self.stop_words = set(stopwords.words('russian'))  # Получение множества русских стоп-слов

    def preprocess_text(self, text):
        """
        Предобрабатывает текст: токенизация, лемматизация и удаление стоп-слов.
        """
        tokens = word_tokenize(text.lower())  # Токенизация текста и приведение к нижнему регистру
        lemmatized_tokens = [self.morph.parse(token)[0].normal_form for token in tokens if token.isalnum()]
        # Лемматизация каждого токена и фильтрация неалфавитных токенов
        filtered_tokens = [token for token in lemmatized_tokens if token not in self.stop_words]
        # Удаление стоп-слов из лемматизированных токенов
        return ' '.join(filtered_tokens)  # Объединение токенов обратно в строку

    def load_from_excel(self, filename, sheetname):
        """
        Загружает данные из Excel файла и возвращает колонку 'tweet'.
        """
        return pd.read_excel(filename, sheet_name=sheetname)['tweet']

    def load_from_excel_and_process(self, filename, sheetname):
        """
        Загружает данные из Excel файла, предобрабатывает текст и возвращает предобработанные данные.
        """
        return list(self.load_from_excel(filename, sheetname).apply(str).apply(self.preprocess_text))

    def load_data(self):
        """
        Загружает и предобрабатывает данные из Excel файлов с негативными и позитивными твитами.
        Сохраняет предобработанные данные в файлы с помощью pickle.
        """
        print('Чтение данных из Excel файлов...')
        dfn = self.load_from_excel('negative_excel.xlsx', 'negativeansi')  # Загрузка негативных данных
        dfp = self.load_from_excel('positive_excel.xlsx', 'positiveansi')  # Загрузка позитивных данных
        print('Предобработка текста...')
        neg_x_arr = list(dfn.apply(str).apply(self.preprocess_text))  # Предобработка негативных данных
        pos_x_arr = list(dfp.apply(str).apply(self.preprocess_text))  # Предобработка позитивных данных
        neg_y_arr = ['negative' for _ in range(len(neg_x_arr))]  # Создание меток для негативных данных
        pos_y_arr = ['positive' for _ in range(len(pos_x_arr))]  # Создание меток для позитивных данных
        x_arr = neg_x_arr + pos_x_arr  # Объединение негативных и позитивных данных
        y_arr = neg_y_arr + pos_y_arr  # Объединение меток
        # Сохранение предобработанных данных и меток в файлы
        with open('xpresave', 'wb') as fp:
            pickle.dump(x_arr, fp)
        with open('ypresave', 'wb') as fp:
            pickle.dump(y_arr, fp)
        return x_arr, y_arr  # Возвращение предобработанных данных и меток

    def load_data_and_split(self):
        """
        Загружает и предобрабатывает данные, затем разделяет их на обучающую и тестовую выборки.
        """
        x_arr, y_arr = self.load_data()  # Загрузка и предобработка данных
        return train_test_split(x_arr, y_arr, test_size=0.2, random_state=42)  # Разделение данных

    def load_presaved_data(self):
        """
        Загружает предобработанные данные из ранее сохраненных файлов.
        """
        with open('xpresave', 'rb') as fp:
            x_arr = pickle.load(fp)  # Загрузка предобработанных данных
        with open('ypresave', 'rb') as fp:
            y_arr = pickle.load(fp)  # Загрузка меток
        return x_arr, y_arr  # Возвращение загруженных данных и меток

    def load_presaved_data_and_split(self):
        """
        Загружает предобработанные данные из ранее сохраненных файлов и разделяет их на обучающую и тестовую выборки.
        """
        x_arr, y_arr = self.load_presaved_data()  # Загрузка предобработанных данных и меток
        return train_test_split(x_arr, y_arr, test_size=0.2, random_state=42)  # Разделение данных
