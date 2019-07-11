from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

class LogisticRegressionWrapper:

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.fit_vectorizer()
        params = self.grid_search()
        self.C = params['C']
        self.penalty = params['penalty']
        self.solver = params['solver']

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=1)
        return X_train, X_test, y_train, y_test

    def fit_vectorizer(self):
        self.vectorizer = CountVectorizer(binary=True)
        self.vectorizer.fit(self.X_train)

    def grid_search(self):
        param_grid ={'penalty' : ['l1', 'l2'],
                     'C' : np.logspace(-4, 4, 20),
                     'solver' : ['liblinear']}

        clf = GridSearchCV(LogisticRegression(), param_grid, cv = 10)
        clf.fit(self.vectorizer.transform(self.X_train), self.y_train)
        return clf.best_params_

    def fit(self):
        self.clf = LogisticRegression(C = self.C, penalty = self.penalty, solver = self.solver)
        self.clf.fit(self.vectorizer.transform(self.X_train), self.y_train)

    def evaluate(self):
        y_pred = self.clf.predict(self.vectorizer.transform(self.X_test))
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print("Accuracy: {}, precision: {}, recall: {} \nConfusion matrix : \n{}".format(acc,prec,rec,cm))
        return acc,prec,rec,cm