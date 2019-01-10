# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:58:31 2019

@author: Ishan Kumar
"""

import numpy as np
import pandas as pd


data=pd.read_csv('training_queries.csv')
data2=pd.read_csv('training_queries_labels.csv')
data3=pd.read_csv('testset.csv')

#Importing Dataset
X=data.iloc[:,1].values
Y=data2.iloc[:,1].values
Z=data3.iloc[:,1].values

#Lemmatizing The Words
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
X=lemmatizer.lemmatize(X)

#Converting to vector
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

X_train = vect.fit_transform(X)
X_test = vect.transform(Z)

#Building the Logisitc Regression Model
from sklearn.linear_model import LogisticRegression
Classifier=LogisticRegression(random_state=0)
Classifier.fit(X_train,Y)

#Predicting Final values
y_pred_class = Classifier.predict(X_test)
