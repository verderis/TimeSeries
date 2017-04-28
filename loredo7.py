#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:04:10 2017

@author__ = 'Fran'

Time series change point detection based on a fully parametrized 
modeling of every date. Online mode, with new observation updating model state
"""

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import pymc as pm
import funciones as f
import pandas as pd
from collections import OrderedDict


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

t = fechas

pixel = 10 #analized pixel
sample_data = evi[pixel,:]

#%% rellenar NaNs
nans, x= nan_helper(sample_data) 
sample_data[nans]= np.interp(x(nans), x(~nans), sample_data[~nans]) 

#%% total data definition
total_data = sample_data
total_doys = doys
total_t = t

samplesPerYear = np.unique(doys).size

#%% data range definition

#rango = range(0,23*3,6)
rango = range(0,total_data.size,23*8)
#rango = range(0,total_data.size,23*2)
#rango = range(0,total_data.size,total_data.size-1)
#rango = range(0, 23 + 23)

reco_all = OrderedDict()
reco_095_all = OrderedDict()
reco_005_all = OrderedDict()
tiempo_all = OrderedDict()

y_all = OrderedDict()
y_095_all = OrderedDict()
y_005_all = OrderedDict()

#observation error = 0.05 EVI units, 
#np.percentile(priorSamp["sigma_err"],5)
#Out[7]: 0.0023692011371226939
#np.percentile(priorSamp["sigma_err"],95)
#Out[8]: 0.15133163074074646

sigma_err_prior = 1/0.05 

for new in rango:
    
    ## analyzed data selection
    sample_data = total_data[:samplesPerYear+new]
    doys = total_doys[:samplesPerYear+new]
    t = total_t[:samplesPerYear+new]
    n_count_data = len(sample_data)
    
    # anual frequency calculation
    anual_freq = {}
    for i in range(samplesPerYear):
        anual_freq[i] = sample_data[doys == doys[i]]
    
    EVI_means = np.zeros(samplesPerYear)
    data1 = pd.DataFrame(anual_freq.items(), columns=['doy', 'EVI'])
    
    k = 0
    for i,j in data1.iterrows():
        EVI_means[i] = j[1].mean()
        k = k+1
        
    data1['EVI_means'] = EVI_means 
         
    ## prior definition for 16 day EVI - pyMC Container
    x = np.empty(samplesPerYear, dtype=object)
    for i in range(samplesPerYear):
        #mean = sample_data[i]
        
        ## uniform prior [0,1]
        x[i] = pm.Uniform('x_%i' % i, lower=0, upper=1)
        
    x = pm.Container(x)
    
    ## prior for observation process, Exp(sigma_err_prior)
    sigma_err = pm.Exponential("sigma_err", sigma_err_prior)
    
    #%% PyMC parameters
    nSamples = 1000
    nChain = 10000
    nBurn = 1000
    
    #%% prior samples
    priorSamp = OrderedDict()
    for rVar in x:
        priorSamp[str(rVar)] = np.asarray([rVar.random() for i in np.arange(nSamples)])
        
    priorSamp["sigma_err"] = np.asarray([sigma_err.random() for i in np.arange(nSamples)])
         
    #%% model definition
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
    
    #%%  reconstruction model definition
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
    
    
    #%% likelihood
    observation = pm.Normal("obs", mu=loredo, tau=1/sigma_err, 
                            value=sample_data, observed=True)
    simulated_data = pm.Normal("sim", mu=loredo, tau=1/sigma_err, 
                            value=sample_data)    
    model = pm.Model([observation, loredo, x, sigma_err, simulated_data])
    
    #%% Montecarlo
    mcmc = pm.MCMC(model)
    mcmc.sample(nChain + nBurn, nBurn, 1)
    
    #%% posterior samples
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
    
    postSamp['sigma_err'] = mcmc.trace('sigma_err')[:]
    
    reco = loredo_det(x_mean, t)
    reco_095 = loredo_det(x_095, t)
    reco_005 = loredo_det(x_005, t)
 
    reco_all[new] = reco
    reco_095_all[new] = reco_095
    reco_005_all[new] = reco_005
    
    ## posterior of reconstructed model trace
    traza_sim=mcmc.trace('sim')[:]
    
    y = np.zeros(total_data.size)
    y_095 = np.zeros(total_data.size)
    y_005 = np.zeros(total_data.size)
    
    for j in range(sample_data.size):
        
        ## posterior of reconstructed model statistics per date
        y[j] = traza_sim[:,j].mean() 
        y_095[j] = np.percentile(traza_sim[:,j],95)
        y_005[j] = np.percentile(traza_sim[:,j],5)
    
    y_all[new] = y 
    y_095_all[new] = y_095
    y_005_all[new] = y_005
    
#%% plotting

N = len(y_005_all.keys()) + 1

#%% reconstruncion using x mean values
#k = 1
#for key in reco_all.keys():
#    fillColor = (0.5*k/N,0.0,0.3)
#    plt.plot(t[:samplesPerYear+key], reco_all[key], alpha=0.3, 
#             linestyle = ':', color=fillColor)
#    plt.fill_between(t[:samplesPerYear+key], reco_005_all[key], 
#                     reco_095_all[key], alpha=0.3, color=fillColor)
#    k = k + 1

#%% reconstruncion using simlated outputs
l = 1
for key in y_005_all.keys():
    fillColor = (0.5*l/N,0.0,0.3) #fill color
    nonZero = y_all[key].nonzero() #nonzero datarange selection
    
    # mean value and confidence interval
    plt.plot(fechas[nonZero], y_all[key][nonZero], alpha=0.7, 
             linestyle = ':', color='blue')
    plt.fill_between(fechas[nonZero], y_005_all[key][nonZero], 
                     y_095_all[key][nonZero], alpha=0.3, color=fillColor)
    
    # breackpoint detection & plotting
    whereUp = (sample_data[nonZero] > y_095_all[key][nonZero]).nonzero()[0]
    whereDown = (sample_data[nonZero] < y_005_all[key][nonZero]).nonzero()[0]
    if len(whereUp) > 0:
        for brk in whereUp:
            plt.plot([t[brk], t[brk]], [0.1, 0.6], color='red', 
                     linestyle = ':')
    if len(whereDown) > 0:
        for brk in whereDown:
            plt.plot([t[brk], t[brk]], [0.1, 0.6], color='red', 
                     linestyle = '-.')
            
    l = l+1

# real data plotting
plt.plot(t, sample_data, color='orange', linestyle = ':') #data

#thick ploting
#for i in range(0,sample_data.size,23):
#    plt.plot([fechas[i],fechas[i]], [0.1, 0.6], color='grey', linestyle = ':', alpha=0.3) #anual period

#plt.legend()
plt.grid(True)
plt.show()
plt.savefig('loredo'+str(new)+'.png')

#%% convergency
#gewekeScores = {}
#for var in x:   
#    gewekeScores[var] = pm.geweke(postSamp[str(var)], first=0.1, 
#                                        last=0.5, intervals=20)
#    
#pm.Matplot.geweke_plot(gewekeScores)
#pm.Matplot.autocorrelation(mcmc)

#%% 95-5 percentil plot
#l = 1
#for key in y_005_all.keys():
#    fillColor = (0.5*l/N,0.0,0.3)
##    plt.plot(fechas, y_all[key], alpha=0.3, linestyle = ':', color=fillColor)
##    plt.fill_between(fechas, y_005_all[key], 
##                     y_095_all[key], alpha=0.3, color=fillColor)
#    plt.plot(fechas, y_095_all[key]-y_005_all[key], color=fillColor, alpha=0.7)
#    l = l+1
#
#plt.legend()
#plt.ylim([0.12,0.4])
#plt.show()

#%% x hist plot
#for rVar in x:
#    plt.hist(postSamp[str(rVar)],histtype='step', normed = True, label='post '+str(rVar))
#    plt.hist(priorSamp[str(rVar)],histtype='step', normed = True, label='prior '+str(rVar))
#plt.legend()
#sigma_trace = mcmc.trace('sigma_err')[:]
#
#plt.hist(postSamp['sigma_err'],histtype='step', normed = True, label='post '+'sigma_err')
#plt.hist(priorSamp['sigma_err'],histtype='step', normed = True, label='prior '+'sigma_err')
#plt.legend()