import string
import numpy as np
import pandas as pd
from config import Config
from scipy.sparse import hstack
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor():
    def __init__(self, config, ngram_c, analyzer_c, max_features_c, ngram_t, analyzer_t, max_features_t, meta):
        """Initialize hyperparameters and load vocabs
        Args:
            config: source csv data path config and global paramaters config
            ngram_c: ngram range for CountVectorizer
            analyzer_c: analyse by 'word' or by 'char' for CountVectorizer
            max_feature_c: max feature for CountVectorizer, None or integer
            ngram_t: ngram range for TfidfVectorizer
            analyzer_c: analyse by 'word' or by 'char' for TfidfVectorizer
            max_feature_c: max feature for TfidfVectorizer, None or integer
            stop_words: stop words or rear words
            meta: whether loading meta data and adding meta level statistic features (T/F)
        """
        self.config = config
        self.ngram_c = ngram_c
        self.analyzer_c = analyzer_c
        self.max_features_c = max_features_c
        self.ngram_t = ngram_t
        self.analyzer_t = analyzer_t
        self.max_features_t = max_features_t
        self.stop_words = 'english'
        self.meta = meta
        

    def load_statistic_features(self, train_x, valid_x):
        """Load statistic features including count and tfidf

        Args:
            train_x: training x dataframe
            valid-x: valid x dataframe
            
        """
        tfidf_vec = TfidfVectorizer(min_df=2, max_features= self.max_features_t, strip_accents='unicode', analyzer=self.analyzer_t,token_pattern=r'\w{1,}',ngram_range=self.ngram_t, use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = self.stop_words)
        full_tfidf = tfidf_vec.fit_transform(self.config.train_df['text'].values.tolist() + self.config.test_df['text'].values.tolist())
        train_tfidf = tfidf_vec.transform(train_x['text'].values.tolist())
        test_tfidf = tfidf_vec.transform(valid_x['text'].values.tolist())

        count_vec = CountVectorizer(max_features= self.max_features_c, analyzer=self.analyzer_c, token_pattern=r'\w{1,}', ngram_range=self.ngram_c, stop_words = self.stop_words)
        full_count = count_vec.fit(self.config.train_df['text'].values.tolist() + self.config.test_df['text'].values.tolist())
        train_count = count_vec.transform(train_x['text'].values.tolist())
        test_count = count_vec.transform(valid_x['text'].values.tolist())

        if self.meta == True:
            train_count_plus_meta = hstack((train_x.iloc[:,-9:].values,train_count))
            test_count_plus_meta = hstack((valid_x.iloc[:,-9:].values,test_count))
            train_tfidf_plus_meta = hstack((train_x.iloc[:,-9:].values,train_tfidf))
            test_tfidf_plus_meta = hstack((valid_x.iloc[:,-9:].values,test_tfidf))

        stat = {}
        stat["full_tfidf"] = full_tfidf
        stat["train_tfidf"] = train_tfidf
        stat["test_tfidf"] = test_tfidf
        stat["full_count"] = full_count
        stat["train_count"] = train_count
        stat["test_count"] = test_count

        if self.meta:
            stat["train_count_plus_meta"] = train_count_plus_meta
            stat["test_count_plus_meta"] = test_count_plus_meta
            stat["train_tfidf_plus_meta"] = train_tfidf_plus_meta
            stat["test_tfidf_plus_meta"] = test_tfidf_plus_meta

        return stat