
import numpy as np
from config import Config
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

class SVMClaasifier():
    def __init__(self, multiclass_logloss, fe, evaluate = True):
        """Initialize a Naive Bayes Classifier
        Args:
            multiclass_logloss: import the loss function
            fe: feature extractor used in this classifier
            evaluate: whether to do cross validation (T/F)
        """
        self.loss = multiclass_logloss
        self.evaluate = evaluate
        self.fe = fe
        
    def train(self, x, y):
        """Train SVM Classifier
        Args:
            x: dataframe contains training info
            y: nmpy array containing labels
        """
        lbl_enc = preprocessing.LabelEncoder()
        y = lbl_enc.fit_transform(y.values)
        # Create a SVM Model
        clf = SVC(C = 0.4, probability = True)
        # Train with cross validation
        if self.evaluate:
            avgLoss = []
            avgProb = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
            for train_index, valid_index in skf.split(x, y):
                train_x, valid_x = x.iloc[train_index, :], x.iloc[valid_index, :]
                train_y, valid_y = y[train_index], y[valid_index]

                # Load features from feature extactor
                stat = self.fe.load_statistic_features(train_x, valid_x)
                # Single value decomposition
                svd = TruncatedSVD(n_components=200)
                svd.fit(stat["full_tfidf"])
                xtrain_svd = svd.transform(stat["train_tfidf"])
                xvalid_svd = svd.transform(stat["test_tfidf"])
                # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
                scl = preprocessing.StandardScaler()
                scl.fit(xtrain_svd)
                xtrain_svd_scl = scl.transform(xtrain_svd)
                xvalid_svd_scl = scl.transform(xvalid_svd)

                clf.fit(xtrain_svd_scl, train_y)
                pred_y = clf.predict(xvalid_svd_scl)
                print("--------------------------------------------------------------")
                print(classification_report(valid_y,pred_y))
                # Predicting with probability
                predictions = clf.predict_proba(xvalid_svd_scl)
                avgProb.append(predictions)
                avgLoss.append(self.loss(valid_y, predictions))
                print("model: SVM")
                print ("logloss: %0.3f " % self.loss(valid_y, predictions))
                print("-------------------------------------------------------")
            print("\033[0;37;41mfinal overall logloss cv: %0.3f \033[0m" % np.mean(avgLoss))
            print("-------------------------------------------------------")
        else:
            stat = self.fe.load_statistic_features(x, x)
            svd = TruncatedSVD(n_components=200)
            svd.fit(stat["full_tfidf"])
            xtrain_svd = svd.transform(stat["train_tfidf"])
            xvalid_svd = svd.transform(stat["test_tfidf"])
            # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
            scl = preprocessing.StandardScaler()
            scl.fit(xtrain_svd)
            xtrain_svd_scl = scl.transform(xtrain_svd)
            xvalid_svd_scl = scl.transform(xvalid_svd)
            # Training
            clf.fit(xtrain_svd_scl, train_y)
        joblib.dump(clf, "./results/classifier_models/svm")
    
    def predict(self, x):
        """Classification for x
        Args:
            x: dataframe contains training info
        Return:
            return an class that this x belonging to.
        """
        clf = joblib.load("./results/classifier_models/svm")
        stat = self.fe.load_statistic_features(x, x)
        predictions = clf.predict_proba(stat["stack_tfidf_plus_meta"])
        return predictions