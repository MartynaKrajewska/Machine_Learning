# -*- coding: utf-8 -*-
"""
Created on Wed May 10 08:38:45 2017

@author: ml383488
"""
# http://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use


# -*- coding: utf-8 -*-
 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import tree
from sklearn.cross_validation import train_test_split
from scipy.io import loadmat
 
 
###############################################################################
# podglądanie obrazków cyfr
# funkcja pomocnicza 
#
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.05)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
 
 
###############################################################################
# wczytywanie danych
dane = loadmat('cyfry.mat')
#przepisanie danych do osobnych tablic:
X = dane['X']
y = dane['y']
for i in range(X.shape[0]):
    X[i,:] = X[i,:]/np.std(X[i,:])
y[np.where(y==10)]=0 # przekodoeanie cyfry 0 tak, żeby w wektorze y też odpowiadąło jej 0 (w oryginalnym zbiorze danych było 10)
 
# wysokość i szerokość obrazka z cyfrą 
h = 20
w = 20
X0=np.zeros((5000,20))
for i in range(X.shape[0]):
    for j in range(20):
        X0[i,j]= np.mean(X[i,20*j:20*(j+1)]) 

print(X0)

###############################################################################
# Wypisz dane o zbiorze cyfr 
print("dane zawierają %d cyfr po %d pixli" % (X0.shape[0], X0.shape[1]))
 
# Pokaż kilka przykładowych cyfr:
#plot_gallery(X0[0:5000:200,:], y[0:5000:200], h, w, n_row=5, n_col=5)
#plt.show()
 
###############################################################################
# Podziel zbiór na dane treningowe i testowe za pomocą funkcji opisanej tu w proporcji 5:1 :
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html
# 
X_train, X_test, y_train, y_test = train_test_split(X0, y, test_size=0.2, random_state=0)
 
 
 
###############################################################################
# 
DEPTH = 10
MAX_FEAT = 100
clf = tree.DecisionTreeClassifier() # instancja klasyfikatora DecisionTreeClassifier
clf = clf.fit(X_train,y_train)# fitowanie do danych treningowych
y_pred = clf.predict(X_train)# predykcja na danych testowych

 
# Pokaż kilka przykładowych klasyfikacji:
print("Klasyfikacja X_train")
#plot_gallery(X_test[0:40,:], y_pred[0:40], h, w, n_row=5, n_col=6)
#plt.show()


print("Klasyfikacja X_test")
y_pred = clf.predict(X_test)# predykcja na danych testowych
#plot_gallery(X_test[0:40,:], y_pred[0:40], h, w, n_row=5, n_col=6)
#plt.show()
 
# uzupełnij miary klasyfikacji
print("wynik F1: ",f1_score(y_test, y_pred, average=None))# uzupełnij
print("confusion matrix:")
print(confusion_matrix(y_test,y_pred))
print("raport klasyfikacji:")
print(classification_report(y_test,y_pred))