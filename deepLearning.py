
"""
Rekkache Amira Meriem 
groupe : 02

"""

import keras as kr 

import pandas as p
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score #évaluer le svm
from seaborn import countplot
from matplotlib.pyplot import figure , show
import matplotlib.pyplot as plt
import numpy as np


mnist=kr.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
c=Counter(y_train)
print(c)




d=p.DataFrame(y_train , columns=["class"]) 
print("\n les observation :")
figure()
countplot(data=d,x='class')
show()


#affichage des images
for i in range(0,50):
    plt.imshow(x_train[i+1],cmap=plt.cm.binary)
print("**********")     

# reshape to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs 0-255 --> 0-1
x_train = x_train / 255
x_test = x_test / 255
################################"
model=kr.Sequential()
model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(128 , activation="relu"))
model.add(kr.layers.Dense(128 , activation="relu"))
model.add(kr.layers.Dense(10 , activation="softmax"))
model.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metric=["accuracy"])
model.fit(x_train , y_train , epochs=3)
#evaluation
scores = model.evaluate(x_test, y_test, verbose=0)
print(scores * 100)
print(100 - scores * 100)

