# -*- coding: utf-8 -*-
#importujemy potrzebne moduły i klasy
import numpy as np
import pylab as py
from svm_modul import *
 
#==================================================================
#                 Program
#==================================================================
 
# wczytywanie danych
dane = np.loadtxt('Dane1.txt') # dane zorganizowane są w trzech kolumnach
N_przyk, N_wej = dane.shape 
X = dane[:,0:2] # pierwsze dwie kolumny to wejście
y = dane[:,2] # trzecia kolumna to etykiety klas
 
# narysujmy te dane
rysujDaneGrup(X, y, marker=('or','xb'), xlabel='x0', ylabel='x1',legend_list=('klasa0','klasa1'))
py.show()
 
# trenujemy model
py.figure(figsize=(10,20))
C= [1,2,5,10,20,30,60,120]
for k,i in enumerate(C):
    py.subplot(8,2,k+1)
    model  = svmTrain(X, y, C[k], kernelFunction = 'linearKernel', tol = 1e-3, max_passes = 20,sigma = 10) 
    py.title(i)
    # {1,2,5,10,20,30,60,120}

    # prezentujemy podział przestrzeni wejść reprezentowany przez model
    rysujDaneGrup(X, y, marker=('or','xb'), xlabel='x0', ylabel='x1',legend_list=('klasa0','klasa1'))
    rysujPodzial(model,X)
py.show()

# im wieksze C tym puntk (0.09, 4) nalezy do swojej klasy

