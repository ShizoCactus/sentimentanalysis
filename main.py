from DataProcessing import DataProcessor
from LogisticRegression import LogisticRegressionModel


def main():
    dp = DataProcessor()
    x_train, x_test, y_train, y_test = dp.load_data()
    for i in range(1, 4):
        model = LogisticRegressionModel(x_train, x_test, y_train, y_test, i)
        model.process()
        model.report()
    for i in range(1, 4):
        model = LogisticRegressionModel(x_train, x_test, y_train, y_test, i, 'tfidf')
        model.process()
        model.report()


main()
