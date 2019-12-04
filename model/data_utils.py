import numpy as np
from config import Config
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
class DataUtils():
    def sent2vec(self, s, config, embeddings_index):
        """Generate sentence level embedding
        Args:
            config: source csv data path config and global paramaters config
        """
        words = str(s).lower()
        words = TreebankWordTokenizer().tokenize(words)
        words = [w for w in words if not w in config.eng_stopwords]
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(embeddings_index[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        if type(v) != np.ndarray:
            return np.zeros(300)
        return v / np.sqrt((v ** 2).sum())