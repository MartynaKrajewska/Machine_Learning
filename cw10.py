# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:55:58 2017

@author: ml383488
"""

from pylab import plot,show,figure,imshow,cm, imread, axis
import numpy as np
import pylab as plt
from scipy.cluster.vq import kmeans,vq
 
im = imread('Skan.png')
# Oryginalny obrazek miał przestrzeń barwną RGB.
# Spłaszczamy przestrzeń barwną obrazka
im = im.mean(axis=2)
#oglądamy obrazek
imshow(im, cmap=cm.gray)
axis('off')
show()
imshow(im, cmap=cm.gray)
axis('off')
show()

data = im[:]
data.shape = 256*256,1 #  zamieniamy rysunek (dwuwymiarowa tablica 256x256) na wektor (o długości 256*256 ) 

K_max = 9
J_inter = np.ones(K_max)*1e16
J_intra = np.zeros(K_max)
centroids =[]
d=1
for K in range(2,K_max):
    trial =0
    while (len(centroids)<K)&(trial<20):
        centroids,J_intra[K] = kmeans(data,K)
        trial+=1
    print ('K: ',K, len(centroids))
    for ki in range(len(centroids)):
        for kj in range(ki):
            print (ki, kj)
            print (centroids[ki])
            print (centroids[kj])
            ################
## dopisz kod obliczający odległość między centrami i oznacz ją d
            d1=np.abs(centroids[ki]-centroids[kj])
            
            if d>d1:
                d=d1
            ################
            # jeśli uzyskana odległość jest mniejsza niż dotychczas zapamiętana to ją zapamiętujemy:
            if J_inter[K]>d:
                J_inter[K]=d
    print (K, J_intra[K],J_inter[K])
    
    
figure(1)
plot(range(2,K_max),J_intra[2:]/J_inter[2:])
K_opt = np.argmin(J_intra[2:]/J_inter[2:])+2
 
print (K_opt)


centroids,J_intra[K] = kmeans(data,K_opt) 
# przypisujemy klasę
idx,_ = vq(data,centroids)

# obraz po algorytmie
idx.shape = 256,256
figure(2)
imshow(idx, cmap=cm.gray)
show()

# obraz przed algorytmem
imshow(im, cmap=cm.gray)
axis('off')
show()

##  Histogram

plt.hist(data, bins=50)
show()

