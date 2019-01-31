import pickle
import requests
import json
import numpy as np
import pandas as pd
import re
import music21
from music21 import *
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def ModelIt(fromUser  = 'Default', fileName = []):
  #load the data
  musicFeat = pd.read_csv('/Users/oana/work/insight/features.csv')
  labels = pd.read_csv('/Users/oana/work/insight/labels1.csv')

  #one big dataframe
  df = pd.concat([musicFeat, labels], axis=1)
  df = df.drop(['Unnamed: 0', 'Tempo'], axis = 1)
  df = df.drop(['Composer', 'Title', 'KeySignature','Instrument','initialTimeSignature', 'timeSignatureChange'], axis = 1)
  #print(df.columns)
  #elim rows with no labels:
  df = df.loc[df['Labels'] != 0]

  #print(df.info())
  musicFeatures = df.drop(['Labels'], axis =1).values
  lbls = df['Labels'].values
  #print(lbls)
  #reduce labels to 3:
  lbls3=[]
  for i in lbls:
        if i == 2 or i == 3:
            x = 1
        elif i == 4 or i == 5:
            x = 2
        elif i == 6 or i == 7 or i==8:
            x = 3
        lbls3.append(x)
  lbls3 = pd.Series(lbls3).values
  #define the inputs for the classifier
  #X_htd = musicFeatures[:,1]
  y = lbls3.reshape(-1,1)
  #X_htd = X_htd.reshape(-1,1)
  #split in test-train sets, classify
  print(musicFeatures.shape)
  print(len(lbls3))
  #raise Exception(lbls3)
  X_train, X_test, y_train, y_test = train_test_split(musicFeatures, lbls3, test_size = 0.25, random_state = 21, stratify=y)

  knn = KNeighborsClassifier(n_neighbors = 12)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  pickle.dump(knn, open('model.pkl','wb'))

  if fromUser != 'Default':
    return knn
  else:
    return 'check your input'
