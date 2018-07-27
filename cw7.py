# -*- coding: utf-8 -*-
 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import tree
from sklearn.cross_validation import train_test_split
from scipy.io import loadmat
from sklearn import svm, datasets



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
 
###############################################################################
# Wypisz dane o zbiorze cyfr 
print("dane zawierają %d cyfr po %d pixli" % (X.shape[0], X.shape[1]))
 
# Pokaż kilka przykładowych cyfr:
plot_gallery(X[0:5000:200,:], y[0:5000:200], h, w, n_row=5, n_col=5)
plt.show()

# podzielić zbiór na dane testowe i treningowe w proporcji 5:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=0)


#stworzyć instancję klasyfikatora SVC, można tu wybrać różne funkcje jądra i parametr regularyzacji

clf=svm.SVC(kernel='rbf',C=1)
clf.fit(X_train,y_train.ravel())
y_pred = clf.predict(X_train)
#clf.(X_train)
print("X_train")
plot_gallery(X_train[0:4000:200,:], y_pred[0:4000:200], h, w, n_row=4, n_col=5)
plt.show()


#clf =svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
#clf.score(X_test,y_test)

#y_pred = clf.fit(X_train,y_train).predict(X_test)


y_pred = clf.predict(X_test)
print("X_test")
plot_gallery(X_train[0:800:40,:], y_pred[0:800:40], h, w, n_row=4, n_col=5)
plt.show()


#
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
print(cnf_matrix)

print(classification_report(y_test,y_pred))













# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py