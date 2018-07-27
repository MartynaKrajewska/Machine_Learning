# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:52:14 2017

@author: ml383488
"""

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq,whiten
 
# generujemy dane: 
# - 150 dwuwymiarowych punktów z rozkładu jednorodnego ze średnią (1,1)
# - 150 dwuwymiarowych punktów z rozkładu jednorodnego ze średnią  (0.5,0.5)
 
data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))
data =  whiten(data)
# policz K-Means dla  K = 2 (2 skupiska)
centroids,_ = kmeans(data,2)
# przypisz wektory wejściowe do skupisk
idx,_ = vq(data,centroids)
 
# narysuj wyniki
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()