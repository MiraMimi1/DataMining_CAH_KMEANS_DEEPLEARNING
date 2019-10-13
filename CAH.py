
"""
@author: Rekkache Amira Meriem

groupe : 02
"""

import os
import pandas as p
import numpy as np
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score #évaluer le svm
from sklearn.model_selection import train_test_split

#Lib CAH
import scipy.cluster.hierarchy as h
from matplotlib import pyplot as p2
from matplotlib import pyplot as plt

####################################
os.getcwd()

print('************************ C A H  **************************************')

#chargement des données
data = p.read_table("File_Features.txt", sep=",", header=0, index_col=31)

#dimension des données
print('\n Dimension des données :')
print(data.shape)

#   Description statistique
print('\n Describe :')
print(data.describe())

#   Descreption graphique croisement deux à deux des variables
print('\n Scatter :')
scatter_matrix(data,figsize =(31,31))

#générer la matrice des liens
print('\n Matrice des lignes :')
m=h.linkage(data, method='ward', metric='euclidean')
print(m)

a=h.cophenet(m)
print(a)

#affichage du dendrogramme
print('************************ Dendrogramme *****************************')
p2.title("TP CAH")  
#matérialisation des 28 classes (hauteur t = 200)
#h.dendrogram(m,labels=data.index ,orientation='top', color_threshold = 200 )
#p2.show()

#découpage à la hauteur t = 200 ==> identifiants de 4 groupes obtenus
groupes_cah = h.fcluster(m,t=200,criterion='distance')
print("\n cluster:")
print(groupes_cah)

#index triés des groupes
idg = np.argsort(groupes_cah)
print("groupe triés :")
print(idg)

#affichage des observations et leurs groupes
print("observation :")
print(p.DataFrame(data.index[idg],groupes_cah[idg]))


