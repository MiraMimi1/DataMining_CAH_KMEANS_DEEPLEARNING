#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rekkache Amira Meriem
groupe :02
"""


import os
import pandas as p
import numpy as np
#from pandas.tools.plotting import scatter_matrix
#from matplotlib import pyplot as p1

#Lib CAH
import scipy.cluster.hierarchy as h
from matplotlib import pyplot as p2
from matplotlib import pyplot as plt

#lib Kmeans
from sklearn import cluster
#librairie pour évaluation des partitions
from sklearn import metrics
####################################

os.getcwd()

print('************************Begin**************************************')

data = p.read_table("File_Features.txt", sep=",", header=0, index_col=31)

#dataF.hist(bins=30 , figsize=(28,15))

#dimension des données
print('********************Dimension des données *************************')
print(data.shape)

#   Description statistique
print('************************Describe***********************************')
#data.describe
print(data.describe())

#   Descreption graphique croisement deux à deux des variables
print('************************ Scatter **********************************')
#p1.scatter_matrix(data,figsize =(31,31))

#générer la matrice des liens
print('************************ Matrice des lignes ***********************')
m=h.linkage(data, method='complete', metric='euclidean')
print(m)
print('*************')
a=h.cophenet(m)
print(a)

#affichage du dendrogramme
print('************************ Dendrogramme *****************************')
p2.title("TP CAH")  
#matérialisation des 28 classes (hauteur t = 200)
h.dendrogram(m,labels=data.index ,orientation='top', color_threshold = 200 )
p2.show()

print('***************************** END *********************************')

#découpage à la hauteur t = 200 ==> identifiants de 4 groupes obtenus
groupes_cah = h.fcluster(m,t=200,criterion='distance')
print(groupes_cah)

#index triés des groupes
idg = np.argsort(groupes_cah)
print(idg)

#affichage des observations et leurs groupes
print(p.DataFrame(data.index[idg],groupes_cah[idg]))


print('************************** K-means *********************************')

#k-means sur les données centrées et réduites

kmeans = cluster.KMeans(n_clusters=27)
kmeans.fit(data)
#index triés des groupes
idk = np.argsort(kmeans.labels_)
#affichage des observations et leurs groupes
print(p.DataFrame(data.index[idk],kmeans.labels_[idk]))
#distances aux centres de classes des observations
print(kmeans.transform(data))

print('*********************** correspondance Kmeans & CAH ***************')
#correspondance avec les groupes de la CAH
crs=p.crosstab(groupes_cah,kmeans.labels_)
print(crs)
#utilisation de la métrique "silhouette"
#faire varier le nombre de clusters de 2 à 10
res = np.arange(27,dtype="double")
for k in np.arange(27):
           km = cluster.KMeans(n_clusters= k+2)
           km.fit(data)
           res[k] = metrics.silhouette_score(data,km.labels_)
print(res)

print('*********************** Graphe ***************')
#graphique
plt.title("Silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,29,1),res)
plt.show()
