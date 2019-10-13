"""
@author: Rekkache Amira Meriem

groupe :02
"""
import pandas as p
import numpy as np
#lib Kmeans
from sklearn import cluster
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score #évaluer le svm

from sklearn.model_selection import train_test_split


print('************************** K-means *********************************')

data = p.read_table("File_Features.txt", sep=",", header=0, index_col=31)

#diviser data en attribut & label 
#x contient tt les colonnes sauf la colonne label & y contient label
X = data.iloc[:,:29]
Y = data.iloc[:,30]

#diviser en trainig & test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)  

#k-means sur les données centrées et réduites

kmeans = cluster.KMeans(n_clusters=27)
kmeans.fit(X_train)
pred=kmeans.predict(X_test)
print(pred)
print("\n matrice de confusion ")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred)) 

#index triés des groupes
idk = np.argsort(kmeans.labels_)
#affichage des observations et leurs groupes
print("observation")
print(p.DataFrame(X_train.index[idk],kmeans.labels_[idk]))
#distances aux centres de classes des observations
print("distance au centre des classes")
print(kmeans.transform(X_train))

