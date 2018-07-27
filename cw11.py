# -*- coding: utf-8 -*-
import numpy as np
 
def g(x):
    y = 1./(1+np.exp(-x))
    return y
 
def g_prim(x):
    y = x*(1-x)
    return y
 
 
#zbiór uczący:
# wejście, 
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1] ])
 
# wyjście            
Y = np.array([[0,1],
              [1,0],
              [1,0],
              [0,1]])


# definiujemy rozmiary sieci:
N_wej = X.shape[1] # 2
N_hid = 3
N_wyj = Y.shape[1] # 
 
# inicjujemy połączenia
# wagi ułożone są tak, że w kolejnych wierszach są kolejne neurony 
# a w kolumnach wagi od konkretnego neuronu 
# to +1 jest wagą dla obciążenia
w_1 = 2*np.random.random((N_hid,N_wej+1)) - 1 # pomiędzy warstwą pierwszą (wejściem) a warstwą ukrytą
w_2 = 2*np.random.random((N_wyj,N_hid+1)) - 1 # ?????????????


for cykl in range(10000):
    bl =0
    D_1 = np.zeros((w_1.shape)) #  tablice akumulujące delty do zmiany wag D_1 i D_2
    D_2 = np.zeros((w_2.shape)) # ?????????????????
 
 
    for i in range(0,4):
        # weźmy przykład i-ty
 
        x = X[i,:].reshape(2,1) # ??????????????
        y = Y[i,:].reshape(2,1) # wektor kolumnowy
 
        # propagacja "w przód"
        a_0 = np.vstack((1,x))  # z warstwy wejściowej (zerowej) wychodzi a_0
 
        z_1 = np.dot( w_1, a_0 )# na warstwe 1 wchodzą iloczyny skalarne 
        a_1 = np.vstack((1,g(z_1))) # dokładamy 1 i dostaję wyjście z warstwy 1
 
        z_2 = np.dot( w_2,a_1 ) # na warstwe 3 wchodzą iloczyny skalarne 
        a_2 = g(z_2)
        if cykl == 10000-1:
            print ('a: ',str(a_2.T))
            print ('y: ',str(y.T))
            
        # propagacja "wstecz"
        d_2 = (a_2-y)*g_prim(a_2) # blad popelniany na warstwie wejsciowej
        d_1 = np.dot(w_2.T, d_2) * g_prim(a_1)
        
 
        # akumulujemy poprawki 
        D_2 +=  np.dot( d_2,a_1.T)
        D_1 +=  np.dot( d_1[1:],a_0.T) # od cyklu ?????????????????????
 
        bl += np.dot(d_2.T,d_2) # ??????????????
 
    eta1 = 0.1
    # uaktualniamy wagi
    w_1 -=  eta1*D_1 
    w_2 -=  eta1*D_2
 
    # wypisujemy info o bledzie
    if (cykl% 1000) == 0:
        print ('bl: ', bl)
        
