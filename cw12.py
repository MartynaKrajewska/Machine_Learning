# -*- coding: utf-8 -*-
import matplotlib
#matplotlib.use('TkAgg')
import numpy as np
import pylab as py
 
class siec(object):
    def __init__(self, X, Y, N_hid=3):
        self.X = X
        self.Y = Y
        self.N_wej = X.shape[1] 
        self.N_wyj = Y.shape[1]
        self.N_hid = N_hid
 
        # inicjujemy polaczenia
        # wagi ułożone są tak, że w kolejnych wierszach są kolejne neurony 
        # a w kolumnach wagi od konkretnego neuronu 
        # to +1 jest wagą dla obciążenia
        self.w_1 = (2*np.random.random((self.N_hid, self.N_wej+1)) - 1)/self.N_wej # pomiędzy warstwą pierwszą (wejściem) a warstwą ukrytą
        self.w_2 = (2*np.random.random((self.N_wyj, self.N_hid+1)) - 1)/self.N_hid
        self.dw1 = np.zeros((self.N_hid, self.N_wej+1))
        self.dw2 = np.zeros((self.N_wyj, self.N_hid+1)) 
 
    def g1(self, x):
        y = 1./(1+np.exp(-x))
        return y   
    def g1_prim(self, x):
        y = x*(1-x)
        return y
    def g2(self, x):
        y = x
        return y   
    def g2_prim(self, x):
        y = 1
        return y
    def get_params(self):
        return np.concatenate((self.w_1.reshape(-1), self.w_2.reshape(-1)),1)
 
    def predict(self, x):
        # propagacja "w przód"
        self.a_0 = np.vstack((1,x))  # z warstwy wejściowej (zerowej) wychodzi a_0
        z_1 = np.dot( self.w_1, self.a_0 )# na warstwe 1 wchodzą iloczyny skalarne 
        self.a_1 = np.vstack((1,self.g1(z_1))) # dokładamy 1 i dostaję wyjście z warstwy 1
        z_2 = np.dot( self.w_2, self.a_1 ) # na warstwe 3 wchodzą iloczyny skalarne 
        self.a_2 = self.g2(z_2)
        return self.a_2
 
    def fit_one_step(self, eta1,eta2):
        self.bl = 0
        D_1 = np.zeros((self.N_hid, self.N_wej+1))
        D_2 = np.zeros((self.N_wyj, self.N_hid+1))
        for i in range(0,self.X.shape[0]):
            # weźmy przykład i-ty        
            x = self.X[i,:].reshape(self.N_wej,1)
            y = self.Y[i,:].reshape(self.N_wyj,1)
            self.a_2 = self.predict(x)
 
            # propagacja "wstecz"
            d_2 = (self.a_2 - y)*self.g2_prim(self.a_2)
            d_1 = np.dot(self.w_2.T, d_2) * self.g1_prim(self.a_1)#z_2
 
            # akumulujemy poprawki 
            D_2 +=  np.dot( d_2, self.a_1.T)
            D_1 +=  np.dot( d_1[1:], self.a_0.T)
 
            self.bl += np.dot(d_2.T,d_2)/self.X.shape[0]
        # uaktualniamy wagi
        self.w_1 -=  eta1*D_1 + eta2*self.dw1
        self.w_2 -=  eta1*D_2+  eta2*self.dw2
        self.dw1  =  eta1*D_1 
        self.dw2  =  eta1*D_2   
        return self.bl
def fun(x):
    return (1+10*x+x**2)/(1+2*x**2)
def gen(ile):
    x = np.sort(5*np.random.rand(ile)).reshape((ile,1))
    y = fun(x).reshape((ile,1))
    y+= 0.05*y*np.random.randn(ile).reshape((ile,1))
    return(x,y)
 
def main(argv=None):  
    #zbiór uczący:
    N_przykladow =37
    X,   Y      = gen(N_przykladow) # przykłady do ciągu uczącego
    X_m, Y_m    = gen(N_przykladow) # przykłady do ciągu monitorującego
    py.figure()
    py.plot(X,Y,'.')
    py.show()
 
    # definiujemy obiekt sieci:
    S = siec( X, Y, N_hid= 7)
 
    # liczba epok uczenia
    N_epochs = 1500
    # inicjuję tablice na ewolucje
    err  = np.zeros(N_epochs) #tablica na błąd zbioru uczącego
    err_m  = np.zeros(N_epochs) #tablica na błąd zbioru monitorującego
    wagi = np.zeros((N_epochs,len(S.get_params()))) #tablica na wagi
 
    eta1 = 0.005   
    eta2 = 0.8                                          
    for cykl in range(N_epochs):
        err[cykl] = S.fit_one_step(eta1,eta2) # wykonaj krok uczenia
        for j, x_j in enumerate(X_m): # liczę średni błąd kwadratowy na zbiorze monitorującym:
            err_m[cykl] += (Y_m[j] - S.predict(x_j) )**2
        err_m[cykl] /=  X_m.shape[0]# normalizuję aby uzyskać średni błąd kwadratowy
        wagi[cykl,:] = S.get_params() #pobieram wagi do zapamiętania
 
    #  rysunki
    py.subplot(2,1,1) # błędów
    py.plot(err,'b',label='zb. uczacy')
    py.plot(err_m,'r',label='zb. monitorujacy')
    py.title(u'błąd')
    py.legend()
    py.ylim([0,3])
 
    py.subplot(2,1,2) #wag
    py.plot(wagi)
    py.title('wagi')
    py.ylim([-3,3])
    py.draw()
 
    # funkcja reprezentowana przez sieć na tle punktów zbioru uczącego i prawdziwej (niezaszumionej) relacji y(x).
    x_testowe = np.linspace(0.1,7,100)
    y_testowe = np.zeros(100)
    for i,x in enumerate(x_testowe):
        y_testowe[i] = S.predict(x)
    # prawdziwa relacja z(x)
    z = fun(x_testowe)
    # rysunki:
    py.figure() 
    py.plot(x_testowe,y_testowe,'r', label='regresja')
    py.plot(x_testowe,z,'b', label='relacja prawdziwa')
    py.plot(X,Y,'bo',label='zb. uczacy')
    py.plot(X_m,Y_m,'mo',label='zb. monitorujacy')
    py.legend()
    py.show()
 
if __name__ == "__main__":
    main()

