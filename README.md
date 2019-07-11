This repository contains work on sentiment analysis permormed on dataset https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
Two approaches were implemented:  Logistic Regression on bag-of-words and simple neural network with LSTM layer.

Optimal parameters for Logistic Regression were selected with Grid Search.

*Data was preprocessed as follows:*
1. whitespaces removed
2. symbols removed
3. stemming with Porter Stemmer(optional)
4. stopwords removed 
5. negative stopwords changed to word 'not'(optional) - to avoid missing negative meaning of the word
6. numbers removed(optional)

File *custom_stopwords.py* contains two lists: stopwords and negative_words, both from nltk.stop_words, but split into two lists for 5th step of data processing.
