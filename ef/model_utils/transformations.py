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

""" Data transformation utilites. """

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from functools import reduce

__author__ = "Mario Juez-Gil"
__copyright__ = "Copyright 2018, Mario Juez-Gil"
__credits__ = ["Mario Juez-Gil", "Cesar Garcia-Osorio", 
               "Álvar Arnaiz-González", "Carlos López-Nozal"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Mario Juez-Gil"
__email__ = "mariojg@ubu.es"
__status__ = "Development"

class ProjectionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, base_projectors, standarizer):
        self.bprojs = base_projectors
        self.standarizer = standarizer

    def fit(self, X, y=None):
        if type(X) is not list:
            X = [X]
        self._init_projectors(X)
        self._fit_transform(X)
        return self

    def transform(self, X):
        if type(X) is not list:
            X = [X]
        check_is_fitted(self, 'projectors')

        pid = 0
        all_transformed = []
        for X_i in X:
            trans = np.array([]).reshape(X_i.shape[0], 0)
            for col in range(X_i.shape[2]):
                col_t = self.projectors[pid].transform(X_i[:,:,col])
                trans = np.concatenate((trans, col_t), axis=1)
                pid += 1
            all_transformed.append(trans)
        X_t = np.concatenate(all_transformed, axis=1)
        return self.standarizer.transform(X_t)

    def fit_transform(self, X, y=None):
        if type(X) is not list:
            X = [X]
        self._init_projectors(X)
        return self._fit_transform(X)

    def _fit_transform(self, X):
        pid = 0
        all_transformed = []
        for X_i in X:
            trans = np.array([]).reshape(X_i.shape[0], 0)
            for col in range(X_i.shape[2]):
                col_t = self.projectors[pid].fit_transform(X_i[:,:,col])
                trans = np.concatenate((trans, col_t), axis=1)
                pid += 1
            all_transformed.append(trans)
        X_t = np.concatenate(all_transformed, axis = 1)
        return self.standarizer.fit_transform(X_t)

    def _init_projectors(self, X):       
        self.projectors = [clone(bproj) for bproj in self.bprojs]
