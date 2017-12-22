#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:04:10 2017

@author__ = 'Fran'

Time series change point detection based on a fully parametrized 
modeling of every date. Online mode, with new observation updating model state

23/11/17
PPC: [('shape and pixel ID', 'break point'), ('0_1', 733420.0)]
Manual: [('shape and pixel ID', 'break point'), ('0_1', 731900.0)]
Manual con error: [('shape and pixel ID', 'break point'), ('0_1', 731890.0)]
"""

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import funciones as f
from collections import OrderedDict
import theano as T
import theano.tensor as Tensor
import os
from datetime import date


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

T.config.profile = True

figsize(12.5, 3.5)
#figsize(8, 3.5)

def matlab2datetime(matlab_datenum):
    """ de Vero"""
    day = date.datetime.fromordinal(int(matlab_datenum))
    dayfrac = date.timedelta(days=matlab_datenum%1) - date.timedelta(days = 366)
    return day + dayfrac

def toNomalDate(fecha):
    out = []
    for t in fecha:
        out.append(str(date.fromordinal(int(t))))
    return np.asarray(out)

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

def prepro(total_data, total_doys, total_t, samplesPerYear):
    sample_data = total_data[:samplesPerYear+new]
    doys = total_doys[:samplesPerYear+new]
    t = total_t[:samplesPerYear+new]
    return sample_data, doys, t 

#%%  reconstruction model definition
def loredo_det(x,t):
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

#%% MCMC parametrs
nSamples = 10000
nBurn = 1000

#definir nombre archivo de entrada
path = '/home/tele/Dropbox/IAFE/Deforestacion/Pruebas_Esteban/Anual/datos/'
#path = '/media/tele/DATA/Dropbox/IAFE/Deforestacion/datos/csv/'

evi, lat, lon, all_t, all_doys = f.cargar_datos(path)

#%% Shapefiles
path_sf = '/home/tele/Dropbox/IAFE/Deforestacion/datos/redaf/muestra5/'
#path_sf = '/media/tele/DATA/Dropbox/IAFE/Deforestacion/datos/redaf/muestra5/'

nombre_sf =  'Coleccion_30_Argentina_1976m5_gg.shp'
sf = f.open_shapefile(path_sf, nombre_sf)

year = 2008
shape_list = f.get_shapes_from_year(sf,year) #levanto las ese anio 

samplePlots = OrderedDict()
shape_ID = 0
for shape in shape_list:
    name = shape_ID
    inside = f.points_inside_shape(shape, lon, lat)
    
    pixelsInside = evi[inside]
    pixelID = 0
    for pixel in pixelsInside:
        samplePlots[str(shape_ID)+'_'+str(pixelID)] = pixel
        pixelID += 1
    shape_ID += 1

#%% output definition 
seriesBreaks = []
seriesBreaks.append(('shape and pixel ID', 'break point'))


#%% Model definition
basic_model = pm.Model()

samplesPerYear = 23 #Number of datapoints per year
sigma_err_prior = 1/0.05 #Estimate of EVI error variance
dummyData = np.zeros([0])

shared_data = T.shared(dummyData, borrow=True) #Sahred data definition - Theano

#%% Model definition
with basic_model:
    ## prior for 15-day EVI, U(0-1)
    x = pm.Uniform("x", 0, 1, shape=samplesPerYear)
    x = Tensor.cast(x, 'float64') #required for 
    
    ## prior for observation process, Exp(sigma_err_prior)
    sigma_err = pm.Exponential("sigma_err", sigma_err_prior)
        
    @T.compile.ops.as_op(itypes=[Tensor.dvector],otypes=[Tensor.dvector]) #original
    def loredo(x = x, profile=True):
        L = shared_data.get_value(borrow=True).shape[0]
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
    loredo.grad = lambda *y: y[0]  # Here's the klutz, llama al grad pero es un bug
    loredo.shape = shared_data.get_value(borrow=True).shape[0]    
           
    ## likelihood
    observation = pm.Normal("obs", mu=loredo(x), tau=1/sigma_err, observed=shared_data)

#%% Main loop

for pixel in list(samplePlots.keys())[1:2]:
        
    print('current pixel=' + str(pixel))
    EVI_data = samplePlots[pixel]
    
    #%% rellenar NaNs
    nans, x= nan_helper(EVI_data) 
    EVI_data[nans]= np.interp(x(nans), x(~nans), EVI_data[~nans]) 
    
    #%% total data definition
    total_data = EVI_data
    total_doys = all_doys
    total_t = all_t
    
    #%% data range definition
    rango = range(0,total_data.size-samplesPerYear) ## todo los datos
#    rango = range(0,2) 
    counter = 0
    
    reco_all = np.zeros(total_data.size-samplesPerYear)
    reco_095_all = np.zeros(total_data.size-samplesPerYear)
    reco_005_all = np.zeros(total_data.size-samplesPerYear)
      
    for new in rango:
        
        ## analyzed data selection
        sample_data, doys, t = prepro(total_data, total_doys, total_t, samplesPerYear)
        n_count_data = len(sample_data)
    #    sample_data.set_value(data_loop)
    
        ## shared data update
        #va el -1 porque no va el ultimo para que el algoritmo no haga trampa
        shared_data.set_value(sample_data[:-1]) 
        print('tamaño datos de entrada: '+ str(shared_data.get_value(borrow=True).shape[0]))
    
        # inference
        with basic_model:    
            step = pm.Metropolis()
            trace = pm.sample(nSamples, step=step)
        burned_trace = trace[nBurn:]
        
    #    simulated = burned_trace['sim'] #Esto no anda
        
        #%% posterior samples    
#        x_trace = burned_trace['x']
#        x_mean = x_trace.mean(axis = 0)
#        x_095 = np.percentile(x_trace, 95, axis = 0)
#        x_005 = np.percentile(x_trace, 5, axis = 0)
#
#        # posterior of reconstructed model
#        reco = loredo_det(x_mean,t)
#        reco_095 = loredo_det(x_095,t)
#        reco_005 = loredo_det(x_005,t)  
#        
#        reco_all[new] = reco
#        reco_095_all[new] = reco_095
#        reco_005_all[new] = reco_005

        ## PPC
        ppc = pm.sample_ppc(trace, model=basic_model)
        obs_ppc = ppc['obs']
        reco = obs_ppc.mean(axis=0)
        reco_095 = np.percentile(obs_ppc, 95, axis=0)
        reco_005 = np.percentile(obs_ppc, 5, axis=0)
        
        reco_all[new] = reco
        reco_095_all[new] = reco_095
        reco_005_all[new] = reco_005
                
        
        #Is the last observed data lower than the 5% percentile of the predictions?        
        if sample_data[-1] - reco_005[-1] < 0: 
            counter += 1
            if counter == 2:
                seriesBreaks.append((pixel, t[-1])) #shape and pixel ID + breakpoint time
                break
        else:
            counter = 0
    
#        print('new date = '+str(new))
        print('number of consecutive breaks = '+str(counter))
             
        #%% every loop plot original
        key = new
        fillColor = (0.5,0.0,0.3) #fill color
    #    nonZero = y.nonzero() #nonzero datarange selection
        
        # mean value and confidence interval
        plt.plot(t[:-1], reco, alpha=0.5, 
                 linestyle = ':', color='red', label='mean')
        plt.fill_between(t[:-1], reco_005, 
                         reco_095, alpha=0.3, color=fillColor,
                         label='CI')
        # real data plotting
        plt.plot(t[:-1], sample_data[:-1], color='green', linestyle = ':', label='MODIS EVI') #data
        plt.plot(t[-1], sample_data[-1] , marker='x', color='green', label='new data') #data
        plt.plot(t[-1], reco[-1], color='red', marker='x', label='estimation') #data
        plt.ylim([0.05, 0.55])
        
        when = date.fromordinal(int(t[-1]))
        
        plt.title('Every loop plot for date =' + str(when))
        plt.legend()
        plt.grid(True)
        plt.show()

    
    
#%% Final plotting

print(seriesBreaks)

N = len(reco_all.keys()) + 1

#%% correlation study

x_trace = burned_trace['x']
corr = []
start = 5
total = 23
x_plot = divmod(np.arange(total)+start, total)[1]
for var in x_plot:
    aux = np.correlate(x_trace[:,start], x_trace[:,var])
    corr.append(aux)

plt.plot(x_plot, np.asarray(corr)/np.asarray(corr).max())
plt.grid(True)

for var in np.arange(5):
    plt.hist(x_trace[:,var], alpha = 0.3, label = 'x_'+str(var), 
             normed=True, histtype = 'stepfilled')
    
plt.grid(True)
plt.legend()
plt.show()

#%% reconstruncion using x mean values
#k = 1
#key = new
#fillColor = (0.5*k/N,0.0,0.3)
#plt.plot(t[:samplesPerYear+key], reco_all[key], alpha=0.6, 
#         linestyle = ':', color=fillColor, label='Reconstructed')
#plt.fill_between(t[:samplesPerYear+key], reco_005_all[key], 
#                 reco_095_all[key], alpha=0.3, color=fillColor)
#
## real data plotting
#plt.plot(t, sample_data, color='green', linestyle = ':', label='MODIS EVI') #data
#plt.legend()
#plt.title('Final reconstruction')
#plt.grid(True)
#plt.show()

#%% convergency
#gewekeScores = pm.diagnostics.geweke(burned_trace)
#
#for var in range(samplesPerYear):   
#    plt.plot(gewekeScores['x'][var][:,0],gewekeScores['x'][var][:,1])
#
#plt.title('Geweke scores')
#plt.grid()
#plt.show()

## Aditional plotting (not used)
#k = 1
#for key in reco_all.keys():
#    fillColor = (0.5*k/N,0.0,0.3)
#    plt.plot(t[:samplesPerYear+key], reco_all[key], alpha=0.3, 
#             linestyle = ':', color=fillColor)
#    plt.fill_between(t[:samplesPerYear+key], reco_005_all[key], 
#                     reco_095_all[key], alpha=0.3, color=fillColor)
#    k = k + 1
#
##%% reconstruncion using simlated outputs
#
#
#key = y_005_all.keys()[-1]
#fillColor = (0.5,0.0,0.3) #fill color
#nonZero = y_all[key].nonzero() #nonzero datarange selection
#
## mean value and confidence interval
#plt.plot(fechas[nonZero], y_all[key][nonZero], alpha=0.7, 
#         linestyle = ':', color='blue', label='mean')
#plt.fill_between(fechas[nonZero], y_005_all[key][nonZero], 
#                 y_095_all[key][nonZero], alpha=0.3, color=fillColor,
#                 label='CI')
#
# breakpoint plotting
#whereUp = (sample_data[nonZero] > y_095_all[key][nonZero]).nonzero()[0]
#whereDown = (sample_data[nonZero] < y_005_all[key][nonZero]).nonzero()[0]

#for tbrk in breakPointsUp:
#    for brk in tbrk:
#        plt.vlines(brk, 0.1, 0.6, colors='green', 
#                 linestyle = ':')
#        when = date.fromordinal(int(brk))
#        plt.text(brk,0.6+0.1*np.random.rand(),str(when))
#        
#
#for tbrk in breakPointsDown:
#    for brk in tbrk:
#        plt.plot([brk, brk], [0.1, 0.6], color='red', 
#                 linestyle = '-.')
#        when = date.fromordinal(int(brk))
#        plt.text(brk,0.6,str(when))

#for tbrk in breakPointsDown:
#    for brk in tbrk:
#brk = breakPointsDown[-1][-1]
#plt.plot([brk, brk], [0.1, 0.6], color='red', 
#         linestyle = '-.')
#when = date.fromordinal(int(brk))
#plt.text(brk,0.6,str(when))
##
#            
#
#plt.savefig('loredo'+str(new)+'.png')



    
#pm.Matplot.autocorrelation(basic_model)
