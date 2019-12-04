import numpy as np
import pandas as pd
from config import Config
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from model.feature_extractor import FeatureExtractor
from model.classifier.fasttext_classifier import FasttextClassifier
from model.classifier.naive_bayes_classifier import NaiveBayesClassifer

def train():
    """Train the final stacking model and generate the csv for submission
    """
    config = Config()
    # Define the parameter used in stacking field.
    n_folds = 5
    shuffle = False
    X = config.train_df
    X_submission = config.test_df
    y = config.train_df["author"]
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    # Define the statergy for StatifiedkFold.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=None)

    # Initiate the feature extractors will be used in Naive Bayes.    
    fe0 = FeatureExtractor(config, (1,2), 'word', None, (1,2), 'word', None, meta = True)
    fe1 = FeatureExtractor(config, (1,2), 'word', None, (1,3), 'word', None, meta = True)
    fe2 = FeatureExtractor(config, (1,2), 'word', None, (1,2), 'char', None, meta = True)
    fe3 = FeatureExtractor(config, (1,2), 'word', None, (1,3), 'char', None, meta = True)
    fe4 = FeatureExtractor(config, (1,2), 'word', None, (1,2), 'word', 100, meta = True)
    fe5 = FeatureExtractor(config, (1,2), 'word', 100, (1,2), 'word', None, meta = False)
    fe6 = FeatureExtractor(config, (1,2), 'word', None, (1,2), 'word', None, meta = False)
    fe7 = FeatureExtractor(config, (1,3), 'word', None, (1,2), 'word', None, meta = False)
    fe8 = FeatureExtractor(config, (1,2), 'char', None, (1,2), 'word', None, meta = False)
    fe9 = FeatureExtractor(config, (1,3), 'char', None, (1,2), 'word', None, meta = False)

    # Initiate base classifiers will be used in stacking.
    clfs = [FasttextClassifier(config, evaluate = True),
            NaiveBayesClassifer(multiclass_logloss, fe0, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe1, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe2, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe3, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe4, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe5, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe6, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe7, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe8, evaluate = False),
            NaiveBayesClassifer(multiclass_logloss, fe9, evaluate = False),]

    print ("Creating train and test sets for stacking.")
    # Dataset_blend_train shape: #ofSamples, #ofClfs, #ofClass.
    dataset_blend_train = np.zeros((X.shape[0], len(clfs), 3))
    # Dataset_blend_test shape: #ofSamples, #ofClfs, #ofClass.
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs), 3))
    
    for j, clf in enumerate(clfs):
        i = 0
        print ("Processing Model ", j)
        # Dataset_blend_test_j shape: #ofSamples, #ofFolds, #ofClass
        dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds, 3))
        for (train, test) in skf.split(X, y):
            print ("Processing Fold", i)
            X_train = X.iloc[train, :]
            y_train = y[train]
            X_test = X.iloc[test, :]
            clf.train(X_train, y_train)
            y_submission = clf.predict(X_test)
            # y_submission shape: #ofsamples, #ofClass
            dataset_blend_train[test, j, :] = y_submission 
            # Shape: #ofSamples, #ofClass
            dataset_blend_test_j[:, i, :] = clf.predict(X_submission)
            i = i + 1
        # Average over folds: dim=1 -> #ofSamples, #ofClass
        dataset_blend_test[:, j, :] = dataset_blend_test_j.mean(1)
        
    print ("Stacking.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train.reshape(-1, len(clfs) * 3), y)
    y_submission = clf.predict_proba(dataset_blend_test.reshape(-1, len(clfs) * 3))
    print ("Saving Results.")
    tmp = np.hstack((X_submission["id"].values.reshape(-1, 1), y_submission))
    res = pd.DataFrame(tmp, columns = ["id", "EAP", "HPL", "MWS"])
    res.to_csv("./results/submission.csv")

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

if __name__ == '__main__':
    train()