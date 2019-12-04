import sys
import numpy as np
import pandas as pd
from config import Config
from model.feature_extractor import FeatureExtractor
from model.classifier.svm_classifier import SVMClaasifier
from model.classifier.fasttext_classifier import FasttextClassifier
from model.classifier.naive_bayes_classifier import NaiveBayesClassifer
from model.classifier.logestic_reg_classifier import LogesticRegClassifier

def main():
    config = Config()
    # Load feature extractors.
    print("Loading 10 feature extractors")
    fe0 = FeatureExtractor(config, (1, 2), 'word', None, (1, 2), 'word', None, meta = True)
    fe1 = FeatureExtractor(config, (1, 2), 'word', None, (1, 3), 'word', None, meta = True)
    fe2 = FeatureExtractor(config, (1, 2), 'word', None, (1, 2), 'char', None, meta = True)
    fe3 = FeatureExtractor(config, (1, 2), 'word', None, (1, 3), 'char', None, meta = True)
    fe4 = FeatureExtractor(config, (1, 2), 'word', None, (1, 2), 'word', 100, meta = True)
    fe5 = FeatureExtractor(config, (1, 2), 'word', 100, (1, 2), 'word', None, meta = False)
    fe6 = FeatureExtractor(config, (1, 2), 'word', None, (1, 2), 'word', None, meta = False)
    fe7 = FeatureExtractor(config, (1, 3), 'word', None, (1, 2), 'word', None, meta = False)
    fe8 = FeatureExtractor(config, (1, 2), 'char', None, (1, 2), 'word', None, meta = False)
    fe9 = FeatureExtractor(config, (1, 3), 'char', None, (1, 2), 'word', None, meta = False)

    # Param for fe: bi_gram + word analyzer + meta features
    NaiveBayesClassifer(multiclass_logloss, fe0, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: tri_gram + word analyzer + meta features
    NaiveBayesClassifer(multiclass_logloss, fe1, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: bi_gram + char analyzer + meta features
    NaiveBayesClassifer(multiclass_logloss, fe2, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: bi_gram + char analyzer + meta features
    NaiveBayesClassifer(multiclass_logloss, fe3, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: bi_gram + word analyzer + 100 hot words + meta features
    NaiveBayesClassifer(multiclass_logloss, fe4, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: bi_gram + word analyzer + 100 hot words
    NaiveBayesClassifer(multiclass_logloss, fe5, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: bi_gram + word analyzer
    NaiveBayesClassifer(multiclass_logloss, fe6, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: tri_gram + word analyzer
    NaiveBayesClassifer(multiclass_logloss, fe7, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: bi_gram + char analyzer
    NaiveBayesClassifer(multiclass_logloss, fe8, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Param for fe: tri_gram + word analyzer
    NaiveBayesClassifer(multiclass_logloss, fe9, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Evaluating the cross validation for FastText Classifier.
    FasttextClassifier(config, evaluate = True)\
    .train(config.train_df, config.train_df["author"])

    # Evaluating the cross validation for Logestic Regression Classifier.
    LogesticRegClassifier(multiclass_logloss, fe0, evaluate = True)\
    .train(config.train_df, config.train_df["author"])
    
    # Evaluating the cross validation for SVM.
    SVMClaasifier(multiclass_logloss, fe0, evaluate = True)\
    .train(config.train_df, config.train_df["author"])
    


def multiclass_logloss(actual, predicted, eps = 1e-15):
    """Multi class version of Logarithmic Loss metric.
        Args:
            actual: Array containing the actual target classes.
            predicted: Matrix with class predictions, one probability per class.
    """
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota    

if __name__ == "__main__":
    main()
