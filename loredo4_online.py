#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:23:06 2017

@author: Fran
Deforestacion online a partir de MODIS

TODO:
"""

__author__ = 'Fran'

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import pymc as pm
import funciones as f
import pandas as pd
from collections import OrderedDict
#from scipy.stats import mode


figsize(8, 7)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


#definir nombre archivo de entrada
path = '/home/tele/Dropbox/IAFE/Deforestacion/datos/csv/'
evi, lat, lon, fechas, doys = f.cargar_datos(path)

timeStep = doys[2]-doys[1] # dias/muestra

pixel = 0 #analized pixel
sample_data = evi[pixel,:]
nans, x= nan_helper(sample_data)
# rellenar NaNs
sample_data[nans]= np.interp(x(nans), x(~nans), sample_data[~nans]) 

samplesPerYear = np.unique(doys).size
t = fechas

# ciclos enteros
#offset = 6
#aniosTotales = 15
#sample_data = sample_data[offset:(aniosTotales*samplesPerYear)+offset]
#doys = doys[offset:(aniosTotales*samplesPerYear)+offset]
#t = t[offset:(aniosTotales*samplesPerYear)+offset]

#new = 1
#sample_data = sample_data[:samplesPerYear+new]
#doys = doys[:samplesPerYear+new]
#t = t[:samplesPerYear+new]

n_count_data = len(sample_data)

# calculo frecuancia anual
anual_freq = {}
for i in range(samplesPerYear):
    anual_freq[i] = sample_data[doys == doys[i]]

EVI_means = np.zeros(samplesPerYear)
data1 = pd.DataFrame(anual_freq.items(), columns=['doy', 'EVI'])

k = 0
for i,j in data1.iterrows():
    EVI_means[i] = j[1].mean()
    k = k+1
    
data1['EVI_means'] = EVI_means #conjeturas de parametros del prior
     
#%% definicion de priors - Containers
x = np.empty(samplesPerYear, dtype=object)
for i in range(samplesPerYear):
    mean = sample_data[i]
#    x[i] = pm.Normal('x_%i' % i, mu=mean, tau=1/0.5)
    x[i] = pm.Uniform('x_%i' % i, lower=0, upper=1)
#    x[i] = pm.Gamma('x_%i' % i, mean, .1)
x = pm.Container(x)
sigma_err = pm.Exponential("sigma_err", 0.01)

#%%parametros de PyMC
nSamples = 1000
nChain = 10000
nBurn = 1000

#%% pido muestras de los prior
priorSamp = OrderedDict()
for rVar in x:
    priorSamp[str(rVar)] = np.asarray([rVar.random() for i in np.arange(nSamples)])
     
##%% definicion del modelo
@pm.deterministic
def loredo(x = x, t = t):
    L = len(t)
    out = np.zeros(L)
    k = 0
    while True:
        if (k+1)*samplesPerYear < L:
            out[k*samplesPerYear:(k+1)*samplesPerYear] = x[0:samplesPerYear]
            k = k+1
        else:
            diff = (k+1)*samplesPerYear - L
            out[k*samplesPerYear:(k+1)*samplesPerYear-diff] = x[0:samplesPerYear-diff]
            break
        
    return out  

## modelo para reconstruccion
def loredo_det(x, t):
    L = len(t)
    out = np.zeros(L)
    k = 0
    while True:
        if (k+1)*samplesPerYear < L:
            out[k*samplesPerYear:(k+1)*samplesPerYear] = x[0:samplesPerYear]
            k = k+1
        else:
            diff = (k+1)*samplesPerYear - L
            out[k*samplesPerYear:(k+1)*samplesPerYear-diff] = x[0:samplesPerYear-diff]
            break
        
    return out  


#%% armo el liklehood
observation = pm.Normal("obs", mu=loredo, tau=sigma_err, 
                        value=sample_data, observed=True)

model = pm.Model([observation, loredo, x])

#%% Aca hace el Montecarlo
mcmc = pm.MCMC(model)
mcmc.sample(nChain + nBurn, nBurn, 1)


#%% pido muestras de la posterior

postSamp = OrderedDict()
x_mean = np.zeros(len(x))
x_095 = np.zeros(len(x))
x_005 = np.zeros(len(x))

k = 0
for rVar in x:
    aux = mcmc.trace(rVar)[:]
    postSamp[str(rVar)] = aux
#    x_mean[k] = aux.mean()
    x_mean[k] = np.median(aux)
    x_095[k] = np.percentile(aux,95)
    x_005[k] = np.percentile(aux,5)
    k = k+1
    
reco = loredo_det(x_mean, t)
reco_095 = loredo_det(x_095, t)
reco_005 = loredo_det(x_005, t)

tiempo = np.asarray(range(len(t)))
plt.plot(t, reco, label='reconstructed model')
plt.plot(t, sample_data, label='MODIS EVI')

plt.fill_between(t, reco_005, reco_095, 
                 alpha=0.5, color="#7A68A6", label = '.95 confidence')
plt.legend()
plt.show()
#plt.plot(tiempo, reco, tiempo, 0.3+0.1*np.sin(np.pi+2*np.pi*tiempo/23))

#plt.hist(postSamp['x_16'],histtype='step', normed = True, label='post')
#plt.hist(priorSamp['x_16'],histtype='step', normed = True, label='prior')
#plt.legend()

#%% convergency
#gewekeScores = {}
#for var in x:   
#    gewekeScores[var] = pm.geweke(postSamp[str(var)], first=0.1, 
#                                        last=0.5, intervals=20)
#    
#pm.Matplot.geweke_plot(gewekeScores)
#pm.Matplot.autocorrelation(mcmc)

##t_masUno = t[-1]
#t_masUno = t[-1]+timeStep
###estimacion no probabilistica
#d_reco = np.zeros(n_count_data)
#
#

##%% estimacion probabilistica t+1
#d_masUno_samples = histPost(t_masUno,postSamp['omega_1'], postSamp['lambda_1'], 
#                            postSamp['B1'], postSamp['B01'])
#
#d_masUno_p = np.mean(d_masUno_samples)
#
#%%graficar
#plt.figure(0)

# datos y modelo
#ax1 = plt.subplot2grid((4,4), (0,0), colspan=3)
#plt.plot(t[:], sample_data[:])
#plt.plot(t[:], d_reco[:])
#plt.xlabel("Time")
#plt.ylabel("Data")
#plt.ylim((sample_data[-1]-0.2,sample_data[-1]+0.2))
#
#ax11 = plt.subplot2grid((4,4), (0, 3))
#plt.hist(d_masUno_samples, histtype='step', bins=30, alpha=0.5,
#          label="post t+1", color="g", orientation='horizontal')
#plt.legend(loc="upper right")
#plt.hlines(d_masUno_p,0,4000,'r',linestyle="--")
#plt.ylim((sample_data[-1]-0.2,sample_data[-1]+0.2))
#
#nVar = len(priorSamp.keys())
#
#i = 1
#j = 0
#for var in priorSamp.keys():
#    if j > 3:
#        j = 0
#        i = i + 1
#    ax2 = plt.subplot2grid((4,4), (i, j))
#    plt.hist(postSamp[var], histtype='step', bins=30, alpha=0.5,
#          label='post '+var, color="r")
#    plt.hist(priorSamp[var], histtype='step', bins=30, alpha=0.5,
#          label='prior ' +var, color="b")
#    plt.legend(loc="upper right")
#    j = j + 1
#
#plt.show()
#
#
