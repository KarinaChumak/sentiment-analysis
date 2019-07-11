import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, precision_score, recall_score
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dropout, Embedding, Dense, LSTM, GRU,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

class NeuralNetWrapper:

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()
        self.fit_tokenizer()

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
        return X_train,X_val,X_test,y_train,y_val,y_test

    def fit_tokenizer(self):
        self.tokenizer =  Tokenizer()
        self.tokenizer.fit_on_texts(self.X_train.values)
        self.max_len = max([len(x.split()) for x in self.X_train] )
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def get_vectors(self, text_df):
        tokens = self.tokenizer.texts_to_sequences(text_df)
        return pad_sequences(tokens, maxlen=self.max_len, padding='post')

    def model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 100, input_length=self.max_len))
        model.add(LSTM(units=64,activation='relu', dropout=0.5))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self):
        self.model = self.model()
        earlystopper = EarlyStopping(monitor='val_loss', patience=115, verbose=1)
        train_vec = self.get_vectors(self.X_train)
        val_vec = self.get_vectors(self.X_val)
        self.model.fit(train_vec, self.y_train, verbose = 0, batch_size=128, epochs=50, validation_data=(val_vec, self.y_val),
                  callbacks=[earlystopper])

    def evaluate(self):
        y_pred = np.array([np.round(x) for x in self.model.predict(self.get_vectors(self.X_test))])
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        print("Accuracy: {}, precision: {}, recall: {} \nConfusion matrix : \n{}".format(acc,prec,rec,cm))
        return acc,prec,rec,cm