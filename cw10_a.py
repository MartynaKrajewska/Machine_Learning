# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:30:06 2017

@author: ml383488
"""

# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
import pylab as py
import random, copy
import numpy as np
import sys
 
 
def pnorm(x, m, s):
    """
    Oblicza gęstość wielowymiarowego rozkładu normalnego dla punktów
    w wektorze x
    Parametry rozkładu :
    m - średnia
    s- macierz kowariancji
    dla zwiększenia czytelności kodu stosujemy typ matrix
    """
    xm = np.matrix(x-m)
    xmt = np.matrix(x-m).transpose()
    for i in range(len(s)): # zabezpieczenie dla większej stabliności numerycznej
        if s[i,i] <= sys.float_info[3]: # min float
            s[i,i] = sys.float_info[3]
    sinv = np.linalg.inv(s)
 
    return (2.0*np.pi)**(-len(x)/2.0)*(1.0/np.sqrt(np.linalg.det(s)))\
            *np.exp(-0.5*(xm*sinv*xmt))
 
def draw_params(t,nbclusters):
        '''funkcja do losowania parametrów początkowych
        t - zbiór treningowy
        '''
        nbobs,nbfeatures = t.shape
        # inicjuje średnie przez losowanie punktu ze zbioru danych
        tmpmu = np.array([t[random.uniform(0,nbobs),:]],np.float64)
        # kowariancje inicjowane są jako macierze diagonalne , wariancja dla każdej cechy inicjowana jest jako wariancja tej cechy dla całego zbioru 
        sigma = np.zeros((nbfeatures,nbfeatures))
        for f in range(nbfeatures):
            sigma[f,f] = np.var(t[:,f])
        #phi inicjujemy tak, że każda składowa mieszanki ma takie samee prawdopodobieństwo
        phi = 1.0/nbclusters
        print ('INIT:',tmpmu,sigma,phi)
        return {'mu': tmpmu,\
                'sigma': sigma,\
                'phi': phi}
 
def plot_gauss(mu,sigma):
    ''' Funkcja rysująca kontury funkcji gęstości prawdopodobieństwa 
       dwuwymiarowego rozkładu Gaussa'''
 
    x = np.arange(-6.0, 6.0001, 0.1)
    y = np.arange(-6.0, 6.0001, 0.1)
    X,Y = np.meshgrid(x, y)
    X.shape = 1,len(x)*len(y)
    Y.shape = 1,len(x)*len(y)
    P = np.vstack((X,Y))
    invS = np.linalg.inv(sigma)
    R = P.T-mu
    z = np.zeros(len(R))
    for i in range(len(R)):
        z[i] = np.exp(-0.5*np.dot( R[i,:].T,np.dot(invS,R[i,:])))
 
    z.shape = len(x),len(y)
    py.contourf(x,y,z,alpha = 0.5)
    py.plot(mu[0],mu[1],'o')
    

def expectation_maximization(t, nbclusters=2, nbiter=3, normalize=False,\
        epsilon=0.001, monotony=False, datasetinit=True):
    """
    t - zbiór treningowy, 
    Każdy wiersz t jest przykładem (obserwacją), każda kolumna to cecha 
    'nbclusters' ilość klastrów, z których budujemy model mieszany
    'nbiter' ilość iteracji
    'epsilon' kryterium zbieżności
 
     Powtórz kroki E i M aż do spełnienia warunku |E_t - E_{t-1}| < ε
 
    Funkcja zwraca parametry modelu (centra i macerze kowariancji Gaussów i ich wagi \phi) oraz 
    etykiety punktów zbioru treningowego oznaczające do którego z Gaussów w modelowanej mieszance należą.
    """
 
    nbobs,nbfeatures = t.shape
 
    ### Opcjonalna normalizacja
    if normalize:
        for f in range(nbfeatures):
            t[:,f] /= np.std(t[:,f])
 
 
    result = {}
    random.seed()
 
    # szykujemy tablice na prawdopodobieństwa warunowe
    Pz = np.zeros((nbobs,nbclusters)) # P(z|x): opisywane równaniami (2) i (3) z wykładu 
    Px = np.zeros((nbobs,nbclusters)) # P(x|z): opisywane równaniem (4)  
 
    # inicjujemy parametry dla każdego składnika mieszankni
    # params będzie listą taką, że params[i] to słownik
    # zawierający parametry i-tego składnika mieszanki
    params = []
    for i in range(nbclusters):
        params.append( draw_params(t,nbclusters) )
 
    old_log_estimate = sys.maxint                   # init
    log_estimate = sys.maxint/2 + epsilon      # init
    estimation_round = 0    
 
    # powtarzaj aż zbiegniesz 
    while (abs(log_estimate - old_log_estimate) > epsilon\
                and (not monotony or log_estimate < old_log_estimate)):
        restart = False
        old_log_estimate = log_estimate   
        ########################################################
        # krok E: oblicz Pz dla każdego przykładu (czyli w oznaczeniach z wykładu w_i^j)
        ########################################################
        # obliczamy prawdopodobieństwa  Px[j,i] = P(x_j|z_j=i)  
        for j in range(nbobs): # iterujemy po przykładach
            for i in range(nbclusters): # iterujemy po składnikach
                Px[j,i] = pnorm(t[j,:], params[i]['mu'], params[i]['sigma']) #  (równanie 4)
 
        #  obliczamy prawdopodobieństwa Pz[j,i] = P(z_j=i|x_j)   
        #  najpierw licznik równania (3)   
        for j in range(nbobs): 
            for i in range(nbclusters):
                Pz[j,i] = Px[j,i]*params[i]['phi']
        #  mianownik równania (3)
        for j in range(nbobs): 
            tmpSum = 0.0
            for i in range(nbclusters):
                tmpSum += Px[j,i]*params[i]['phi']
        # składamy w całość Pz[j,i] = P(z_j=i|x_j)
            Pz[j,:] /= tmpSum
 
        ###########################################################
        # krok M: uaktualnij paramertry (sets {mu, sigma, phi}) #
        ###########################################################
        #print "iter:", iteration, " estimation#:", estimation_round,\
        #            " params:", params
        for i in range(nbclusters):
            print ("------------------")
            # parametr phi: równanie (6)
            Sum_w = np.sum(Pz[:,i])
            params[i]['phi'] = Sum_w/nbobs
            if params[i]['phi'] <= 1.0/nbobs:           # restartujemy jeśli zanika nam któraś składowa mieszanki
                restart = True                          
                print ("Restarting, p:",params[i]['phi']) 
                break
            print ('i: ',i,' phi: ', params[i]['phi'])
            # średnia: równanie (7)
            m = np.zeros(nbfeatures)
            for j in range(nbobs):
                m += Pz[j,i]*t[j,:]
            params[i]['mu'] = m/Sum_w
            print ('i: ',i,' mu: ', params[i]['mu'])
 
            # macierz kowariancji: równanie (8)
            s = np.matrix(np.zeros((nbfeatures,nbfeatures)))
            for j in range(nbobs):
                roznica = np.matrix(t[j,:]-params[i]['mu'])
                s += Pz[j,i]*(roznica.T*roznica)
            params[i]['sigma'] = s/Sum_w
 
            print (params[i]['sigma'])
 
            ### Testujemy czy składniki się nie sklejają i w razie potrzeby restartujemy
            if not restart:
                restart = True
                for i in range(1,nbclusters):
                    if not np.allclose(params[i]['mu'], params[i-1]['mu'])\
                    or not np.allclose(params[i]['sigma'], params[i-1]['sigma']):
                        restart = False
                        break
            if restart:                
                old_log_estimate = sys.maxint                 # init
                log_estimate = sys.maxint/2 + epsilon    # init
                params = [draw_params(t,nbclusters) for i in range(nbclusters)] # losujemy nowe parametry startowe
                print ('RESTART')
                continue
 
 
            ####################################
            # liczymy estymatę log wiarygodności: równaie (1)  #
            ####################################
            log_estimate = np.sum([np.log(np.sum(\
                    [Px[j,i]*params[i]['phi'] for i in range(nbclusters)]))\
                    for j in range(nbobs)])
            print ("(EM) poprzednia i aktualna estymata log wiarygodności: ",\
                    old_log_estimate, log_estimate)
            estimation_round += 1
        ##########################
        #  rysujemy aktualny stan modelu
        ##########################
        py.ioff()
        py.clf()
        py.ion()
        for i in range(nbclusters):
            plot_gauss(np.array(params[i]['mu']),np.array(params[i]['sigma']))
        py.plot(x[:,0],x[:,1],'g.')
        py.axis('equal')
        py.draw()
 
 
        # Pakujemy wyniki
        quality = -log_estimate
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['params'] = copy.deepcopy(params)
            result['clusters'] = [[o for o in range(nbobs)\
                    if Px[o,c] == max(Px[o,:])]\
                    for c in range(nbclusters)]
    return result



##########################################
#   PROGRAM
#########################################
 
 
# robimy mieszankę dwóch gaussów:
#parametry rozkładu
# wektor średnich:
mu1 = [-2,-3] 
# macierz kowariancji:
Sigma1 = np.array([[1, 0.5],
                  [0.5, 1]])
# generujemy dane: 
x1 = np.random.multivariate_normal(mu1, Sigma1, 150) #
mu2 = [-0.5,2] 
# macierz kowariancji:
Sigma2 = np.array([[3, 0.5],
                  [0.5, 1]])
# generujemy dane: 
x2 = np.random.multivariate_normal(mu2, Sigma2, 150) #
# łączymy x1 i x2 aby otrzymac jeden zbiór
x = np.vstack((x1,x2))
py.plot(x[:,0],x[:,1],'g.')
py.axis('equal')
py.show()
py.figure()
res = expectation_maximization(x, nbclusters=2, nbiter=3, normalize=False,\
        epsilon=0.001, monotony=False, datasetinit=True)
py.ioff()
py.show()
# wypisz parametry
print ('Dopasowany model: ')
print (res['params'])

def prob_mix(params, x):
    '''params - parametry dopasowanego gaussowskiego modelu mieszanego
    x - punkt wejścowy,
 
    funkcja zwraca gestość prawdopodobieństwa, dla x w rozkładzie mieszanym
    '''
    prob = 0
    for i in range(len(params)):
        prob+= pnorm(x, params[i]['mu'], params[i]['sigma']) * params[i]['phi']
 
 
    return prob
#---------------- przykładowe użycie: ----------------
x=(6,-4)
print ('P(x=(',str(x),')):', prob_mix(res['params'], x))