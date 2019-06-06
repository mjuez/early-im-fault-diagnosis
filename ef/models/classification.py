#!/usr/bin/env python
# Copyright (C) 2018  Mario Juez-Gil <mariojg@ubu.es>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Classifiers built with scikit-learn.

The main idea of the classifiers proposed here, is to first perfom data
transformation (through PCA for instance), and then, to use a simple
classifier (Decision Tree) with the new data.

Various metrics will be used to caracterize the classifiers:
 - F1 MACRO
 - F1 MICRO
 - ACCURACY
 - LABEL RANKING LOSS
 - HAMMING LOSS
 - ZERO ONE LOSS
"""

import numpy as np
from ef.model_utils.transformations import ProjectionTransformer as PT
from sklearn.externals.joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, \
                            label_ranking_loss, zero_one_loss
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

__author__ = "Mario Juez-Gil"
__copyright__ = "Copyright 2018, Mario Juez-Gil"
__credits__ = ["Mario Juez-Gil", "Cesar Garcia-Osorio", 
               "Álvar Arnaiz-González", "Carlos López-Nozal"]
__license__ = "GPLv3"
__version__ = "0.2"
__maintainer__ = "Mario Juez-Gil"
__email__ = "mariojg@ubu.es"
__status__ = "Development"

class ProjectedBaseClassifier(BaseEstimator):
    """ Projected Base Classifier.

    This sklearn classifier performs a projection transformation to the input
    data (i.e. PCA), and then builds a given base classifier (i.e. 
    DecisionTree) that works with the transformed data.

    Args:
        projector:  object for transforming input data using it's projections
                    (i.e. PCA)
        classifier: base classifier (i.e. DecisionTree)
        split_cols: if transforming each column of the input separately.
    
    Attributes:
        projector:   object for transforming input data using it's projections
                     (i.e. PCA)
        classifier:  base classifier (i.e. DecisionTree)
        split_cols:  if transforming each column of the input separately.

    """

    def __init__(self, projectors, classifier):
        self.projectors = projectors
        self.classifier = classifier

    def fit(self, X, y):
        self.pt_ = PT(self.projectors, StandardScaler())
        X_transformed = self.pt_.fit_transform(X, y)
        self.classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['pt_'])
        X_transformed = self.pt_.transform(X)
        return self.classifier.predict(X_transformed)

def _eval_classifier(X, Y, classifier, n_kfolds, n_reps, random_state, n_cpus):
    """ Evaluates a classifier using KFolds.

    This generic function is for evaluating a classifier using KFolds.
    Some evaluation metrics used are f1 score, accuracy or hamming loss.

    Args:
        X:                input data.
        Y:                output labels.
        classifier:       classifier to evaluate.
        n_kfolds:         number of cross validation folds.
        n_reps:           number of repetitions. 
        random_state:     k folds algorithm random state.
    
    Returns:
        measures:         classifier evaluation metrics measures.

    """

    def split_training_test(X_in, train_idxs, test_idxs):
        X_train, X_test = (None, None)
        if type(X_in) is list:
            X_train, X_test = ([], [])
            for X_i in X_in:
                X_train.append(X_i[train_idxs])
                X_test.append(X_i[test_idxs])
        else:
            X_train = X_in[train_idxs]
            X_test = X_in[test_idxs]
        return tuple((X_train, X_test))
    
    with Parallel(n_cpus, 'loky') as parallel:
        def eval_folds():
            def eval_fold(train_idxs, test_idxs):
                measures = [0, 0, 0, 0, 0, 0]
                fold_classifier = clone(classifier)
                X_train, X_test = split_training_test(X, train_idxs, test_idxs)
                Y_train, Y_test = Y[train_idxs], Y[test_idxs]

                fold_classifier.fit(X_train, Y_train)
                Y_pred = fold_classifier.predict(X_test)
                measures[0] = f1_score(Y_test, Y_pred, average='macro')
                measures[1] = f1_score(Y_test, Y_pred, average='micro')
                measures[2] = accuracy_score(Y_test, Y_pred)
                measures[3] = label_ranking_loss(Y_test, Y_pred)
                measures[4] = hamming_loss(Y_test, Y_pred)
                measures[5] = zero_one_loss(Y_test, Y_pred)
                return np.array(measures)

            kf = KFold(n_kfolds, True, random_state)
            folds = kf.split(X[0] if type(X) is list else X)
            m = parallel(delayed(eval_fold)(tri, tei) for tri, tei in folds)
            return np.array(m)

        m = [eval_folds() for _ in range(n_reps)]
        return np.around(np.array(m), 4).tolist()

def build_pca_dt(X, Y, n_pc_comps = [], n_kfolds = 10, n_reps = 10, 
    random_state = None, n_cpus = -2):
    pcas = []
    for n_components in n_pc_comps:
        pcas.append(PCA(n_components=n_components))
    dt = DecisionTreeClassifier()
    pbc = ProjectedBaseClassifier(pcas, dt)
    pbc_all = clone(pbc).fit(X,Y)

    return pbc_all, _eval_classifier(X, Y, pbc, n_kfolds, n_reps, random_state,
                                    n_cpus)

def build_pca_rf(X, Y, n_pc_comps = 10, n_estimators=10, n_kfolds = 10,  
    n_reps = 10, random_state = None, split_cols = True, n_cpus = -2):
    pca = PCA(n_components=n_pc_comps)
    rf = RandomForestClassifier(n_estimators=n_estimators)
    pbc = ProjectedBaseClassifier(pca, rf, split_cols)
    pbc_all = clone(pbc).fit(X,Y)

    return pbc_all, _eval_classifier(X, Y, pbc, n_kfolds, n_reps, random_state,
                                    n_cpus)
