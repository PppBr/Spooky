import numpy as np
from config import Config
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

class LogesticRegClassifier():
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
        self.meta_loaded = fe.meta

    def train(self, x, y):
        """Train Naive Bayes Classifier
        Args:
            x: dataframe contains training info
            y: nmpy array containing labels
        """
        # Mapping authors to author ids
        lbl_enc = preprocessing.LabelEncoder()
        y = lbl_enc.fit_transform(y.values)
        # Create a Logestic Regression Model
        clf = LogisticRegression(C=1.0)
        # Training with cross validation
        if self.evaluate:
            avgLoss = []
            avgProb = []
            skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 2018)
            for train_index, valid_index in skf.split(x, y):
                train_x, valid_x = x.iloc[train_index, :], x.iloc[valid_index, :]
                train_y, valid_y = y[train_index], y[valid_index]
                
                # Load features from feature extactor
                stat = self.fe.load_statistic_features(train_x, valid_x)
                if not self.meta_loaded:
                    train_x =  stat["train_count"]
                    valid_x = stat["test_count"]
                else:
                    train_x = stat["train_tfidf_plus_meta"]
                    valid_x = stat["test_tfidf_plus_meta"]

                clf.fit(train_x, train_y)
                pred_y = clf.predict(valid_x)
                print("--------------------------------------------------------------")
                print(classification_report(valid_y,pred_y))
                # Predicting with probability
                predictions = clf.predict_proba(valid_x)
                avgProb.append(predictions)
                avgLoss.append(self.loss(valid_y, predictions))
                print("model: Logestic regression")
                print ("logloss: %0.3f " % self.loss(valid_y, predictions))
            print("\033[0;37;41mfinal overall logloss cv: %0.3f \033[0m" % np.mean(avgLoss))
            print("-------------------------------------------------------")
        else:
            if not self.meta_loaded:
                x =  stat["train_count"]
            else:
                x = stat["train_tfidf_plus_meta"]
            # Training
            clf.fit(x, y)
        # Model dumping
        joblib.dump(clf, "./results/classifier_models/nbmc")
        
    def predict(self, x):
        """Classification for x
        Args:
            x: dataframe contains training info
        Return:
            return an class that this x belonging to.
        """
        clf = joblib.load("./results/classifier_models/nbmc")
        # Features loading
        stat = self.fe.load_statistic_features(x, x)
        if not self.meta_loaded:
            x =  stat["test_count"]
        else:
            x = stat["test_tfidf_plus_meta"]
        # Predicting
        predictions = clf.predict_proba(x)
        return predictions
