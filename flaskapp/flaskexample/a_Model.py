import pickle
import requests
import json
import numpy as np
import pandas as pd
import re
import os
import music21
from music21 import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def ModelIt(fromUser  = 'Default', fileName = []):
  #load the data
  musicFeat = pd.read_csv('/home/ubuntu/flaskapp/features6.csv')
  labels = pd.read_csv('/home/ubuntu/flaskapp/labels3.csv')

  #one big dataframe
  df = pd.concat([musicFeat, labels], axis=1)
  #Drop weird first column, drop the composer and title fields
  df = df.drop(['Unnamed: 0'], axis = 1)
  df = df.drop(['Composer', 'Title'], axis = 1)

  #elim rows with no labels:
  df = df.loc[df['Labels'] != 0]
  #define the feature dataframe
  musicFeatures = df.drop(['Labels'], axis =1).values
  #define the labels series
  lbls = df['Labels'].values

  #reduce labels to 3:
  lbls3=[]
  for i in lbls:
      if i == 2 or i == 3:
          x = 1
      elif i == 4 or  i == 5:
          x = 2
      elif i==8 or i == 6 or i == 7 :
          x = 3
      lbls3.append(x)
  lbls3 = pd.Series(lbls3).values
  #define the inputs for the classifier
  y = lbls.reshape(-1,1)
  X_train, X_test, y_train, y_test = train_test_split(musicFeatures, lbls, test_size = 0.2, random_state = 21, stratify=y)

  knn = KNeighborsClassifier(n_neighbors = 4)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  pickle.dump(knn, open('model.pkl','wb'))

  if fromUser != 'Default':
    return knn
  else:
    return 'check your input'
