# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:55:49 2022

@author: austi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle5 as pickle

import warnings # filtering out messy warning messages
warnings.filterwarnings("ignore")

df = pd.read_csv('Fraud.csv')
df.head()

df = df.drop(columns = ['nameOrig', 'nameDest']).reset_index(drop = True)
df = df.dropna(axis=0,how = 'any')
df = pd.concat([df, pd.get_dummies(df['type'])], axis = 1).drop(columns = ['type'])
X = df.drop(columns = ['isFraud'])
y = df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)

print('Training set shape: ', X_train.shape)
print('Testing set shape: ', X_test.shape)

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 5, 5), random_state=5)

nn.fit(X_train, y_train)

with open('nnmodel.pkl','wb') as f:
    pickle.dump(nn,f)

print('Training accuracy: ', nn.score(X_train, y_train))
print('Testing accuracy: ', nn.score(X_test, y_test))