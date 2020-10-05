
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



class DataHandler:

    def __init__(self, path, data):
        self.data = data
        self.path = path


    def read_data(self):
        self.data = pd.read_csv(self.path, ',')




    def data_statistics(self):
        data_head = self.data.head(5)
        classes_volume = self.data.iloc[:, 3].value_counts()

        return data_head, classes_volume


    def train_test_split(self):
        self.read_data()
        return train_test_split(self.data['text'], self.data.label, test_size=0.2, random_state=6)




    def text_to_vector(self):
        x_train, x_test, y_train, y_test = self.train_test_split()
        vectorizer = TfidfVectorizer(stop_words='english', max_df= 0.7)
        train_vectors = vectorizer.fit_transform(x_train)
        test_vectors = vectorizer.transform(x_test)

        return train_vectors, test_vectors, y_train, y_test
































# dhandler = DataHandler('/home/mohsen/datascience/fake-news/data/news.csv', pd.DataFrame())
# dhandler.train_test_split()
# head_data , classes_count = dhandler.data_statistics()

