#!/usr/bin/env python3
#
# Take a folder and setup the targets for predicting y-y_{linear}
#

import sys, os
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.linear_model import SGDRegressor
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import logging
from netCDF4 import Dataset
import skimage

# Add s4pi module to patch
_S4PI_DIR = os.path.abspath(__file__).split('/')[:-3]
_S4PI_DIR = os.path.join('/',*_S4PI_DIR)
sys.path.append(_S4PI_DIR+'/4piuvsun/')
from s4pi.data.preprocessing import loadAIAMap


# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


def getEVEInd(data_root,split):
    xind = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,-1]
    
    df_indices = pd.read_csv(data_root+split+'.csv')
    yind = np.asarray(df_indices[df_indices.columns[-1]]).astype(int)

    return yind, xind


def handleStd(index_aia_i):

    divide = 4

    AIA_sample = np.asarray([np.expand_dims(divide*divide*skimage.transform.downscale_local_mean(loadAIAMap(aia_file).data, (divide, divide)), axis=0) for aia_file in index_aia_i], dtype = np.float64 )
    X = np.nanmean(AIA_sample,axis=(1,2,3))
    X = np.concatenate([X,np.nanstd(AIA_sample,axis=(1,2,3))],axis=0)
    return np.expand_dims(X,axis=0)

def save_prediction(eve_data, line_indices, prediction, data_root, split, debug=False):

    matches = pd.read_csv(data_root+split+'.csv')
    if debug:
        matches = matches.loc[0:4,:]

    # open database
    netcdfDB = Dataset(data_root + 'EVE_linear_pred_' + split + '.nc', "w", format="NETCDF4")
    netcdfDB.title = f'{split} EVE observed and predicted spectral irradiance for specific spectral lines using a linear model'
    netcdfDB.split = split + ' set'

    # Assemble variables
    eve_date = eve.variables['isoDate'][:][matches['eve_indices'].values]
    eve_jd = eve.variables['julianDate'][:][matches['eve_indices'].values]
    eve_logt = eve.variables['logt'][:][line_indices]
    eve_name = eve.variables['name'][:][line_indices]
    eve_wl = eve.variables['wavelength'][:][line_indices]
    eve_irr = eve.variables['irradiance'][:][matches['eve_indices'].values,:][:,line_indices]

    # Create dimensions
    isoDate = netcdfDB.createDimension("isoDate", None)
    name = netcdfDB.createDimension("name", eve_name.shape[0])

    # Create variables and atributes
    isoDates = netcdfDB.createVariable('isoDate', 'S2', ('isoDate',))
    isoDates.units = 'string date in ISO format'

    julianDates = netcdfDB.createVariable('julianDate', 'f4', ('isoDate',))
    julianDates.units = 'days since the beginning of the Julian Period (January 1, 4713 BC)'

    names = netcdfDB.createVariable('name', 'S2', ('name',))
    names.units = 'strings with the line names'

    wavelength = netcdfDB.createVariable('wavelength', 'f4', ('name',))
    wavelength.units = 'line wavelength in nm'

    logt = netcdfDB.createVariable('logt', 'f4', ('name',))
    logt.units = 'log10 of the temperature'

    irradiance = netcdfDB.createVariable('irradiance', 'f4', ('isoDate','name',))
    irradiance.units = 'spectal irradiance in the specific line (w/m^2)'

    pred_irradiance = netcdfDB.createVariable('pred_irradiance', 'f4', ('isoDate','name',))
    pred_irradiance.units = 'predicted spectal irradiance using a linear model in the specific line (w/m^2)'

    # Intialize variables
    isoDates[:] = eve_date
    julianDates[:] = eve_jd 
    names[:] = eve_name
    wavelength[:] = eve_wl
    logt[:] = eve_logt
    irradiance[:] = eve_irr 

    netcdfDB.close()



def getXy(eve_data, data_root, split, debug=True):

    matches = pd.read_csv(data_root+split+'.csv')

    if debug:
        matches = matches.loc[0:4,:]

    y = eve_data[matches['eve_indices'].values,:]

    Xs = []
    fnList = []

    aia_columns = [col for col in matches.columns if 'AIA' in col]
    
    for index, row in tqdm(matches.iterrows()):
        fnList.append(row[aia_columns].tolist())

    LOG.info('Start process map')

    Xs = process_map(handleStd, fnList, chunksize=5)

    X = np.concatenate(Xs,axis=0)
   
    mask = y < 0
    y[mask] = 0
 
    return X, y, mask

def addOne(X):
    return np.concatenate([X,np.ones((X.shape[0],1))],axis=1)

def getResid(y,yp,mask,flare=None,flarePct=0.975):
    resid = np.abs(y-yp)
    resid = resid / np.abs(y)
    resid[mask] = np.nan
    return np.nanmean(resid,axis=0)

def fitSGDR_Huber(X,Y,maxIter=10,epsFrac=1.0,logalpha=-4):

    # Remove non-finite points
    finite_mask = np.sum(np.isfinite(Y),axis=1)==14
    X = X[finite_mask,:]
    Y = Y[finite_mask,:]

    finite_mask = np.sum(np.isfinite(X),axis=1)==9
    Y = Y[finite_mask,:]
    X = X[finite_mask,:]

    alpha = 10**logalpha
    K = Y.shape[1]
    models = []
    for j in range(K): 
        model = SGDRegressor(loss='huber',alpha=alpha,max_iter=maxIter,epsilon=np.std(Y[:,j])*epsFrac,learning_rate='invscaling',random_state=1,fit_intercept=False)
        model.fit(X,Y[:,j])
        models.append(model)
    return models

def applySGDmodel(X,models):
    yp = []
    for j in range(len(models)): 
        yp.append(np.expand_dims(models[j].predict(X),axis=1))
    return np.concatenate(yp,axis=1)

def cvSGDH(XTr,YTr,XVa,YVa,maskVa):

    # Remove non-finite points
    finite_mask = np.sum(np.isfinite(YTr),axis=1)==14
    XTr = XTr[finite_mask,:]
    YTr = YTr[finite_mask,:]

    finite_mask = np.sum(np.isfinite(XTr),axis=1)==9
    YTr = YTr[finite_mask,:]
    XTr = XTr[finite_mask,:]


    finite_mask = np.sum(np.isfinite(YVa),axis=1)==14
    XVa = XVa[finite_mask,:]
    YVa = YVa[finite_mask,:]

    finite_mask = np.sum(np.isfinite(XVa),axis=1)==9
    YVa = YVa[finite_mask,:]
    XVa = XVa[finite_mask,:]


    bestPerformance, bestP, bestA = np.inf, 1, 0
    print("CV'ing huber epsilon, regularization")
    for p in range(1,10,1):
        for a in range(-5,1):
            model = fitSGDR_Huber(XTr,YTr,maxIter=10,epsFrac=1.0/p,logalpha=a)
            
            YTrp = applySGDmodel(XTr,model)
            YVap = applySGDmodel(XVa,model)
            residVa = getResid(YVa,YVap,maskVa)
            perf = np.mean(residVa)
            print("a = 10e%d, eps = %f => %f" % (a, 1.0 / p, perf))
            if perf < bestPerformance:
                bestPerformance, bestP, bestA = perf, p, a

    print("Best a = 10e%d, eps = %f" % (bestA,1.0/bestP))
    model = fitSGDR_Huber(XTr,YTr,maxIter=100,epsFrac=1.0/bestP,logalpha=bestA)
    YTrp = applySGDmodel(XTr,model)
    YVap = applySGDmodel(XVa,model)
    W = np.concatenate([np.expand_dims(m.coef_,axis=0) for m in model],axis=0)
    return W

def getNormalize(XTr):

    # Remove non-finite elements
    finite_mask = np.sum(np.isfinite(XTr),axis=1)==8
    XTr = XTr[finite_mask,:]

    mu = np.nanmean(XTr,axis=0)
    sig = np.nanstd(XTr,axis=0)
    sig[sig==0] = 1e-8

    return mu, sig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True)
    parser.add_argument('--divide',dest='divide',default=4,required=True, type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    data_root = args.base

    # Load nc file
    LOG.info('Loading EVE_irradiance.nc')
    eve = Dataset(args.base + 'EVE_irradiance.nc', "r", format="NETCDF4")
    
    eve_data = eve.variables['irradiance'][:]
    line_indices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,14])
    eve_data = eve_data[:,line_indices]

    #get the data
    debug=True

    XTr, YTr, maskTr = getXy(eve_data, data_root, "train", debug=debug)
    XVa, YVa, maskVa = getXy(eve_data, data_root, "val", debug=debug)
    XTe, ___, ______ = getXy(eve_data, data_root, "test", debug=debug)

    np.savez_compressed("%s/mean_std_feats.npz" % data_root,XTr=XTr,XVa=XVa,XTe=XTe)

    mu, sig = getNormalize(XTr)

    XTr = addOne((XTr-mu) / sig)
    XVa = addOne((XVa-mu) / sig)
    XTe = addOne((XTe-mu) / sig)

    model = cvSGDH(XTr,YTr,XVa,YVa,maskVa)

    #Predictions = X*W'
    YTrp = np.dot(XTr,model.T)
    YVap = np.dot(XVa,model.T) 
    YTep = np.dot(XTe,model.T)

    #these are the new targets
    diffTr = YTr - YTrp; diffTr[maskTr] = 0
    diffVa = YVa - YVap; diffVa[maskVa] = 0

    save_prediction(eve, line_indices, YTrp, data_root, 'train', debug=debug)
    save_prediction(eve, line_indices, YVap, data_root, 'val', debug=debug)
    save_prediction(eve, line_indices, YTep, data_root, 'test', debug=debug)

    eve.close()


    
    # #update EVE
    # EVE = np.load(EVE_path)
    # updates = [("train",diffTr),("val",diffVa)]
    # for phaseName,newVals in updates:
    #     yind, xind = getEVEInd(data_root, phaseName)
    #     for yii,yi in enumerate(yind):
    #         EVE[yi,xind] = newVals[yii,:]


    # #new statistics
    # residualMean = np.mean(diffTr,axis=0)   
    # residualStd = np.std(diffTr,axis=0)   
 
    # np.save("%s/eve_residual_mean_14ptot.npy" % data_root, residualMean)
    # np.save("%s/eve_residual_std_14ptot.npy" % data_root, residualStd)

    # #rescale targets
    # EVE *= 100

    # #Save the new target and the model
    # np.save("%s/irradiance_30mn_residual_14ptot.npy" % data_root, EVE)
    # np.savez_compressed("%s/residual_initial_model.npz" % data_root,model=model,mu=mu,sig=sig)

    
