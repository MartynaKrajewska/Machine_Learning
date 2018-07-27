# -*- coding: utf-8 -*-
#  http://scikit-learn.org/stable/tutorial/basic/tutorial.html
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py


# importujemy biblioteki 
from sklearn import svm
from pylab import  show, imshow, subplot, title,cm,axis,matshow,colorbar, plot,figure
import numpy as np
from numpy.random import permutation
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix,  classification_report, f1_score
from sklearn.feature_selection import RFE
from sklearn.cross_validation import train_test_split
 
# wczytywanie danych
dane = loadmat('cyfry.mat')
m,n = dane['X'].shape #ilość przykładów 
 
# wyłuskujemy dane
X = dane['X']
y = dane['y'].ravel()
 
#normalizujemy obrazki
for i in range(X.shape[0]):
    X[i,:] = X[i,:]/np.std(X[i,:])
 
y[np.where(y==10)]=0 # przekodowanie cyfry 0 tak, żeby w wektorze y też odpowiadało jej 0 (w oryginalnym zbiorze danych było 10)
 
 
# prezentacja co 50-tej cyfry ze zbioru
for i in range(100):
    subplot(10,10,i+1)
    C = X[50*i,:].reshape(20,20)
    imshow(C.T,cmap=cm.gray)
    axis('off')
    title(str(y[50*i]))
show()
 
# dzielimy dane na testowe i treningowe w proporcji 1:4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
 
# wytwarzamy instancję klasyfikatora SVC z liniowym jądrem i parametrem regularyzacji =1
clf=svm.SVC(kernel='linear',C=1)

# uczymy klasyfikator na zbiorze treningowym
clf.fit(X_train,y_train.ravel())
 
# obliczamy predykcję i miary klasyfikacji dla zbioru testowego
y_pred = clf.predict(X_train)
 
 
# teraz badamy jakość cech
# wytwarzamy instancję obiektu RFE dla naszego modelu, docelową ilością cech będzie 1, krok 5, żeby było widać co się dzieje włączmy verbose = 1
 
rfe = RFE(clf,1, 5, verbose =1 )
# uczymy obiekt rfe
rfe = rfe.fit(X_train,y_train)
 
# narysujmy rangi 
plot(rfe.ranking_) # to co widać to wartość rangi przypisana dla każdego z 400 pixli obrazka. Czym niższa ranga tym bardziej ważny jest dany pixel. 
 
# żeby lepiej sobie wyobrazić jak rozkładają się rangi w różnych rejonach obrazków możemy je przeformatować do rozmiaru obrazka i wyświetlić w skali kolów
figure()
matshow(rfe.ranking_.reshape(20,20))
colorbar()
title("Ranking of pixels with RFE")
show()
 
# żeby wiedzieć na jakim poziomie odciąć ilość cech dobrze jest posłużyć się miarami jakości predykcji np. F1
# proszę przygotować fragment kodu, który iteracyjnie będzie dopasowywał modele, kolejno do coraz mniejszej liczby optymalnych cech. Dla tak dopasowanych modeli obliczał wybraną miarę i zapamiętywał ją, tak aby po zakończeniu iteracji można było wyświetlić wykres zależności tej miary od ilości cech
f1_list =[]
for i in range(400,200,-5):
    rfe = RFE(clf, i , 1, verbose =1 )
    # uczymy obiekt rfe
    rfe = rfe.fit(X_train,y_train)
    y_pred = rfe.predict(X_test)

    f1_list.append(f1_score(y_test, y_pred, average=None))
    print(f1_list[-1])
    
plot(range(400,200,-5),f1_list)
show()
    
 
# po znalezieniu optymalnej liczby cech dopasuj model z tą właśnie ilością cech i wyświetl maskę za pomocą  której wybieramy istotne cechy do klasyfikacji
#...
# wyświetl co 50-tą cyfrę wyciętą za pomocą tej maski
#...