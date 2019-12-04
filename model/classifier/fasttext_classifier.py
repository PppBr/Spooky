import keras
import string
import numpy as np
import pandas as pd
import keras.backend as K
from sklearn import metrics
from scipy.sparse import vstack
from scipy.sparse import hstack
from sklearn import preprocessing
from collections import defaultdict
from keras.models import Sequential
from keras_preprocessing import text
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, GlobalAveragePooling1D, Embedding

class FasttextClassifier():
    def __init__(self, config, evaluate = True):
        """Initialize hyperparameters and load vocabs
        """
        self.config = config
        self.min_count = 2
        self.max_len = 256
        self.embedding_dims = 20
        self.epochs = 25
        self.batch_size = 16
        self.evaluate = evaluate

    def train(self, x, y):
        """Launch FastText Classifier training.
        Args:
            x: dataframe contains training info.
            y: nmpy array containing labels.
        """
        df = x
        a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
        y = np.array([a2c[a] for a in y])
        y = to_categorical(y)
        # Appending ngram feature at the end of each sentence.  
        docs = self.create_docs(df)
        # Initiate tokenizer for encoding
        tokenizer = Tokenizer(lower=True, filters='')
        tokenizer.fit_on_texts(docs)
        # Dropping the raw words (count < 2)
        num_words = sum([1 for _, v in tokenizer.word_counts.items()\
        if v >= self.min_count])
        tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')
        tokenizer.fit_on_texts(docs)
        self.tokenizer = tokenizer

        # encoding for sentences
        docs = tokenizer.texts_to_sequences(docs)
        # chunking the over length sentence into max_len
        docs = pad_sequences(sequences=docs, maxlen=self.max_len)
        input_dim = np.max(docs) + 1
        # training with cress validation
        if self.evaluate:
            x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size = 0.2)
            model = self.create_model(input_dim)
            hist = model.fit(x_train, y_train,
                            batch_size = self.batch_size,
                            validation_data = (x_test, y_test),
                            epochs = self.epochs,
                            callbacks = [EarlyStopping(patience=2, monitor='val_loss')])
            self.model = model
        else:
            model = self.create_model(input_dim)
            model.fit(docs, y,
                        batch_size = self.batch_size,
                        epochs = self.epochs,
                        callbacks = [EarlyStopping(patience = 2, monitor = 'val_loss')])
            self.model = model

    def predict(self, x):
        """Classification for x.
        Args:
            x: dataframe contains data to be classify.
        Return:
            return an class that x belonging to.
        """
        docs = self.create_docs(x)
        docs = self.tokenizer.texts_to_sequences(docs)
        docs = pad_sequences(sequences = docs, maxlen = self.max_len)
        y = self.model.predict_proba(docs)
        return y
    
    def create_model(self, input_dim, embedding_dims = 20, optimizer='adam'):
        """Launch a FastText 3 layer model.
        Args:
            input_dim: dimension of the input data (largest word id).
            embedding_dims: the out data dimension from embedding layer.
            optimizer: model optimizer.
        Return:
            return a FastText model.
        """
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
        return model        

    def preprocess(self, text):
        """Text pre-processing.
        Args:
            text: input text string.
        Return:
            return a text string after pre-processing.
        """
        text = text.replace("' ", " ' ")
        signs = set(',.:;"?!')
        prods = set(text) & signs
        if not prods:
            return text

        for sign in prods:
            text = text.replace(sign, ' {} '.format(sign) )
        return text

    def create_docs(self, df, n_gram_max=2):
        """Appending ngram sentences at the end of each sentence in df.
        Args:
            df: DataFrame contains corpus.
        Return:
            return the docs with ngram level sentences.
        """
        def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q) - n + 1):
                    ngrams.append('--'.join(q[w_index : w_index+n]))
            return q + ngrams

        docs = []
        for doc in df.text:
            doc = self.preprocess(doc).split()
            docs.append(' '.join(add_ngram(doc, n_gram_max)))
        return docs