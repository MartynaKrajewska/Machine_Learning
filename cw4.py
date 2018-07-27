# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def plot_gauss(mu,sigma,xx,yy):
    ''' Funkcja rysująca kontury funkcji gęstości prawdopodobieństwa 
       dwuwymiarowego rozkładu Gaussa'''
 
    XX = np.c_[xx.ravel(), yy.ravel()]  
    R = XX - mu 
    invS = np.linalg.inv(np.diag(sigma))
    z = np.zeros(len(R))
    for i in range(len(R)):
        z[i] = np.exp(-0.5*np.dot( R[i,:].T,np.dot(invS,R[i,:])))
    z.shape = xx.shape
    plt.contourf(xx,yy,z,alpha = 0.5)
    plt.plot(mu[0],mu[1],'o')
    
    
    
#ładujemy dane
iris = datasets.load_iris() #https://en.wikipedia.org/wiki/Iris_flower_data_set
 
# zapoznajemy się z tymi danymi
print (iris['DESCR'])
# rysujemy zależniści między cechami
# przygotowujemy własną mapę kolorów
color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.5, .5, 0)}
# wytwarzamy wektor, który każdemu wierszowi w tabeli danych przypisze kolor odpowiadający gatunkowi irysa
colors = [color_map[y] for y in iris.target]
plt.figure(1)
plt.title(u'rozkłady cech w klasach')
for i, name in enumerate(iris['feature_names']):
    for j, name in enumerate(iris['feature_names']):
        plt.subplot(4,4,i*4+j+1)
        plt.scatter(iris.data[:,i],iris.data[:,j],c = colors)
 
# wybieramy cechy 2 i 3 i normalizujemy je
X = np.zeros((iris.data.shape[0],2))
X[:,0] = (iris.data[:,2] - np.mean(iris.data[:,2]))/np.std(iris.data[:,2])
X[:,1] = (iris.data[:,3] - np.mean(iris.data[:,3]))/np.std(iris.data[:,3])  
plt.figure(2)
plt.scatter(X[:,0],X[:,1],c = colors)  
plt.title('Wybrane cechy po normalizacji')
plt.show()    
    
#########################################################
 
gnb = GaussianNB()
y_pred=gnb.fit(X, iris.target).predict(X) # stwórz instancję klasyfikatora  Gaussian Naive Bayes 
 # dofituj parametry klasyfikatora 
 
# przedstaw rozkłady Gaussa, które zostały dopasowane do danych, skorzystaj z funkcji plot_gauss()
# średnie tych rozkładów są w gnb.theta_
# standardowe odchylenia są w gnb.sigma_
# przygotowanie siatki na której będą rysowane kontury Gaussów


x_min, x_max = -3,3
y_min, y_max = -3,3
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

plt.figure(4)
for i in range(3):
    plot_gauss(gnb.theta_[i,:],gnb.sigma_[i,:],xx,yy)  #  plot_gauss(mu, sigma, xx, yy)
# dorzućmy do rysunku jeszcze oryginalne dane

plt.scatter(X[:,0],X[:,1],c = colors)
plt.title(u'Rozklady Gaussa dopasowane do danych')
plt.show()        
 
# rysowanie wyników klasyfikacji             
# przekształcamy siatkę w macierz dwukolumnową - kolumny odpowiadają cechom
XX = np.c_[xx.ravel(), yy.ravel()]       
# dla każdego punktu siatki oblicz predykcję klasyfikatora  


Z = gnb.predict(XX)  # klasyfilikacja kazdego punktu 

# te predykcje narysujemy w przestrzeni cech za pomocą funkcji  plt.contourf 
plt.figure(3)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired) #plt.contourf(X, Y, Z, 10,#[-1, -0.1, 0, 0.1],#alpha=0.5,cmap=plt.cm.bone,origin=origin)
# i dorzucamy oryginalne punkty
plt.scatter(X[:,0],X[:,1],c = colors)
plt.title(u'Podział przestrzeni cech na klasy')
plt.show()


 
# Teraz zajmiemy się ewaluacją dopasowanego modelu. Skorzystamy z
# http://scikit-learn.org/stable/modules/model_evaluation.html
# upewnij się, że dokładnie rozumiesz co zwracają te funkcje
# porównaj z definicjami z wykładu 
# http://haar.zfb.fuw.edu.pl/edu/index.php/Uczenie_maszynowe_i_sztuczne_sieci_neuronowe/Wykład_Ocena_jakości_klasyfikacji
print("classification report:")
print(classification_report(iris.target, y_pred)) # ????
print("confusion matrix:")
print(confusion_matrix(iris.target, y_pred))
