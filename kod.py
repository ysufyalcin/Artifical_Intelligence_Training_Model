# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:34:54 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Veri setini oluşturma
dataset=pd.read_csv('veriseti4.csv')

X=dataset.iloc[:,3:-1 ].values

y=dataset.iloc[:,-1].values

# Sayısal olmayan verileri sayısal(nümerik) veriye dönüştürme

from sklearn.preprocessing import LabelEncoder


labelencoder_1=LabelEncoder()

X[:,1]=labelencoder_1.fit_transform(X[:,1])


labelencoder_2=LabelEncoder()

X[:,2]=labelencoder_2.fit_transform(X[:,2])

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

transformer = make_column_transformer(
    (OneHotEncoder(drop='first'), [1,2]),
    remainder='passthrough')

X = transformer.fit_transform(X)

#Eğitim ve Test olarak veri setini bölme
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=26)

#Özellik ölçeklendirme
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Yapay Sinir Ağını Oluştur

import keras
from keras.models import Sequential
from keras.layers import Dense

sınıflandırıcı=Sequential()

#Girdi Katmanı ve 1. Gizli Katman
sınıflandırıcı.add(Dense(units=6,kernel_initializer='uniform', activation='relu',input_dim=11))

#2. Gizli Katman
sınıflandırıcı.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))

#Çıktı Katman
sınıflandırıcı.add(Dense(units=1,kernel_initializer='uniform',   activation='sigmoid'))
#çoklu sınıflandırmada softmax, ikili sınıflandırmada sigmoid aktivasyon fonksiyonu kullanılır.

sınıflandırıcı.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

sınıflandırıcı.fit(X_train,y_train,batch_size=10,epochs=10)
#batch_size kaç örnekte bir ağırlıklar güncellenecek
#epoch kaç tur boyunca eğitim devam edecek

sınıflandırıcı.evaluate(X_test,y_test,batch_size=10)

y_pred=sınıflandırıcı.predict(X_test)

y_pred2=(y_pred>0.5)






















