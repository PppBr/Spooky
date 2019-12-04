import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

class Config():
    def __init__(self):
        """Initialize hyperparameters and load vocabs
        """
        self.meta_loaded = True
        # Source data file config and load
        self.train_df = pd.read_csv("./data/train.csv")
        self.test_df = pd.read_csv("./data/test.csv")
        self.full_df = pd.read_csv("./data/fulltext.csv")
        # External data config
        self.eng_stopwords = set(stopwords.words("english"))
        self.punctuation = string.punctuation
        self.ascii_printable = string.printable
        # Proj config
        pd.options.mode.chained_assignment = None
        # Load meta features
        self.load()

    def load(self):
        """Loads meta features:
        num_words / num_unique_words / num_chars / num_word_upper
        / num_stopwords / num_punctuations / num_words_title
        / mean_word_len
        """
        # Upper case to lower case + remove punctuation [19579, 1]
        self.train_df['text_cleaned'] = self.train_df['text']\
        .apply(lambda x: self.clean_text(x))
        self.test_df['text_cleaned'] = self.test_df['text']\
        .apply(lambda x: self.clean_text(x))

        # total num of words in each sentence [19579, 1]
        self.train_df["num_words"] = self.train_df["text"]\
        .apply(lambda x: len(str(x).split()))
        self.test_df["num_words"] = self.test_df["text"]\
        .apply(lambda x: len(str(x).split()))

        # total num of words in each sentence (distinct) [19579, 1]
        self.train_df["num_unique_words"] = self.train_df["text"]\
        .apply(lambda x: len(set(str(x).split())))
        self.test_df["num_unique_words"] = self.test_df["text"]\
        .apply(lambda x: len(set(str(x).split())))

        # total num of chars in each sentece
        self.train_df["num_chars"] = self.train_df["text"]\
        .apply(lambda x: len(str(x)))
        self.test_df["num_chars"] = self.test_df["text"]\
        .apply(lambda x: len(str(x)))

        # num of non ascii chars in each sentence
        self.train_df["num_non_ascii"] = self.train_df["text"]\
        .apply(lambda x: len([w for w in str(x).split() if w not in self.ascii_printable]))
        self.test_df["num_non_ascii"] = self.test_df["text"]\
        .apply(lambda x: len([w for w in str(x).split() if w not in self.ascii_printable]))

        # total num of Upper words in each sentence
        self.train_df["num_words_upper"] = self.train_df['text']\
        .apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        self.test_df["num_words_upper"] = self.test_df['text']\
        .apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

        # total num of stopwords in each sentence 
        self.train_df["num_stopwords"] = self.train_df["text_cleaned"]\
        .apply(lambda x: len([w for w in str(x).lower().split() if w in self.eng_stopwords]))
        self.test_df["num_stopwords"] = self.test_df["text_cleaned"]\
        .apply(lambda x: len([w for w in str(x).lower().split() if w in self.eng_stopwords]))

        # total num of punctuations in each sentence
        self.train_df["num_punctuations"] = self.train_df["text"]\
        .apply(lambda x: len([w for w in str(x) if w in string.punctuation]))
        self.test_df["num_punctuations"] = self.test_df["text"]\
        .apply(lambda x: len([w for w in str(x) if w in string.punctuation]))

        # total num of capital words in the file
        self.train_df["num_words_title"] = self.train_df["text_cleaned"]\
        .apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
        self.test_df["num_words_title"] = self.test_df["text_cleaned"]\
        .apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
        
        # mean length of a word in the file
        self.train_df["mean_word_len"] = self.train_df["text_cleaned"]\
        .apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        self.test_df["mean_word_len"] = self.test_df["text_cleaned"]\
        .apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    def clean_text(self, x):
        tmp = x.lower()
        for p in self.punctuation:
            tmp = tmp.replace(p, '')
        return tmp