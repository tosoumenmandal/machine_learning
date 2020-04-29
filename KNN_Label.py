# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:30:17 2020

@author: Soumen
"""
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

en= preprocessing.LabelEncoder()
weather_encoded= en.fit_transform(weather)
print(weather_encoded)

temp_encoded= en.fit_transform(temp)
print(temp_encoded)

label= en.fit_transform(play)
print(label)

predictors= list(zip(weather_encoded, temp_encoded))
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit (predictors, label)
predicted_outcome= knn.predict([[0,2]])
print(predicted_outcome)
