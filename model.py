
from sklearn.linear_model import PassiveAggressiveClassifier
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from data_handler import DataHandler

class Model:

    def getData(self):
        data = DataHandler('data/news.csv', pd.DataFrame())
        return data


    def applyModel(self):
        train_vectors, test_vectors, y_train, y_test = self.getData().text_to_vector()
        model = PassiveAggressiveClassifier(max_iter=50)
        model.fit(train_vectors, y_train)
        predict_model = model.predict(test_vectors)
        accuracy = accuracy_score(y_test, predict_model)
        print(f'Accuracy: {round(accuracy * 100, 2)}%')
        print("Confusion Matrix for model is: ")
        print(confusion_matrix(y_test, predict_model, labels=['FAKE', 'REAL']))


Model().applyModel()
