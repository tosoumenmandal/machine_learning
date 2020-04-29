# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:27:32 2020

@author: Soumen
"""
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

wine= datasets.load_wine()
"""print(wine.feature_names)
print(wine.target_names)
print(wine.data.shape)
print(wine.data[0:5])
print(wine.target.shape)
print(wine.target)"""

def predict_score(k):
    X_train, X_test, y_train, y_test= train_test_split(wine.data, wine.target, test_size=0.3)
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred= knn.predict(X_test)
    print('Accuracy for K=',k,':',metrics.accuracy_score(y_test, y_pred))

predict_score(5)
predict_score(7)
