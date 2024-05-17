from DataProcessing import DataProcessor
from LogisticRegression import LogisticRegressionModel


def main():
    dp = DataProcessor()  # Создание объекта DataProcessor для предобработки данных
    # Загрузка и разделение данных на обучающую и тестовую выборки
    # x_train, x_test, y_train, y_test = dp.load_data_and_split()
    x_train, x_test, y_train, y_test = dp.load_presaved_data_and_split()  # Загрузка предварительно сохраненных данных и их разделение

    # Запуск моделей логистической регрессии для униграмм, биграмм и триграмм
    for i in range(1, 4):
        model = LogisticRegressionModel(x_train, x_test, y_train, y_test, i)  # Создание модели для i-грамм
        model.process()  # Векторизация данных и обучение модели
        model.report()  # Вывод отчета о классификации

    # Запуск моделей логистической регрессии с TF-IDF векторизацией для униграмм, биграмм и триграмм
    for i in range(1, 4):
        model = LogisticRegressionModel(x_train, x_test, y_train, y_test, i, 'tfidf')  # Создание модели для i-грамм с использованием TF-IDF
        model.process()  # Векторизация данных и обучение модели
        model.report()  # Вывод отчета о классификации


main()  # Вызов основной функции
