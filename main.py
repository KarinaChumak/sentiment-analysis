import pandas as pd
from logistic_regression import LogisticRegressionWrapper
from nn import NeuralNetWrapper
from data_processing import load_process_data

datasets = ['amazon', 'imdb','yelp']
for ds in datasets:
    print("******************{}*********************".format(ds))
    data = load_process_data(ds, drop_numbers=True, stem=False, replace_negatives=True)
    text, labels = data['processed_text'], data['label']

    log_regr = LogisticRegressionWrapper(text, labels)
    log_regr.fit()
    log_regr.evaluate()

    nn = NeuralNetWrapper(text, labels)
    nn.fit()
    nn.evaluate()

