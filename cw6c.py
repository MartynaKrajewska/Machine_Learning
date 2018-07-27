# -*- coding: utf-8 -*-
#importujemy potrzebne moduły i klasy
import numpy as np
import pylab as py
from svm_modul import *
 
#==================================================================
#                 Program
#==================================================================
 
# wczytywanie danych
dane = np.loadtxt('Dane3.txt') # dane zorganizowane są w trzech kolumnach
N_przyk, N_wej = dane.shape 
X = dane[:,0:2] # pierwsze dwie kolumny to wejście
y = dane[:,2] # trzecia kolumna to etykiety klas
 
# narysujmy te dane
rysujDaneGrup(X, y, marker=('or','xb'), xlabel='x0', ylabel='x1',legend_list=('klasa0','klasa1'))
py.show()
 

# trenujemy model
#sigma = [0.1,0.2,0.4,0.8,1,2,4,8]

# prezentujemy podział przestrzeni wejść reprezentowany przez model
zakresC = np.logspace(np.log2(0.1),np.log2(100),8, base=2)
print(zakresC)
    
# wczytywanie danych
dane = np.loadtxt('dane3.txt') # dane zorganizowane są w trzech kolumnach
N_przyk, N_wej = dane.shape 
X = dane[:,0:2] # pierwsze dwie kolumny to wejście
y = dane[:,2] # trzecia kolumna to etykiety klas
 
#podział na zbiór uczący i testujący
grupa0, = np.where(y==-1)
grupa1, = np.where(y==1)
 
# mieszamy kolejność indexów
np.random.shuffle(grupa0)
np.random.shuffle(grupa1)
 
# kopiujemy dane do zbioru uczącego (pierwsze 75% grupy0 i grupy1)
Xu = X[np.concatenate((grupa0[0: int(0.75*len(grupa0))],grupa1[0:int(0.75*len(grupa0))]))]
yu = y[np.concatenate((grupa0[0: int(0.75*len(grupa0))],grupa1[0:int(0.75*len(grupa0))]))]
# kopiujemy dane do zbioru testowego (końcowe 25% grupy0 i grupy1)
Xt = X[np.concatenate((grupa0[int(0.75*len(grupa0)):], grupa1[int(0.75*len(grupa0)):]))]
yt = y[np.concatenate((grupa0[int(0.75*len(grupa0)):], grupa1[int(0.75*len(grupa0)):]))]
 
 
# narysujmy te dane
 
rysujDaneGrup(Xu, yu, marker=('xr','xb'), xlabel='x0', ylabel='x1',legend_list=('klasa0','klasa1'))
rysujDaneGrup(Xt, yt, marker=('or','ob'), xlabel='x0', ylabel='x1',legend_list=('klasa0_test','klasa1_test'))
py.show()

model  = svmTrain(Xu, yu, C=10, kernelFunction = 'gaussianKernel', tol = 1e-3, max_passes = 20,sigma = 0.5)
TPR = np.sum(yt == svmPredict(model,Xt))/float(len(yt))
rysujDaneGrup(X, y, marker=('or','xb'), xlabel='x0', ylabel='x1',legend_list=('klasa0','klasa1'))
rysujPodzial(model,X)

py.show()
#Proszę napisać program, który
#
# * skanuje przestrzeń (C,sigma): C w zakresie od 0.1 do 100, sigma w zakresie od 0.1 do 10. Do wygenerowania zakresu ze skalą logarytmiczną można wykorzystać np. takie polecenie: zakresC = np.logspace(np.log2(0.1),np.log2(100),8, base=2)
# * znajduje najlepsze parametry
# * rysuje podział przestrzeni dla najlepszych parametrów.