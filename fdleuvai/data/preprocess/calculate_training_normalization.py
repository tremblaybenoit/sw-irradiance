#!/usr/bin/env python
#Make normalization constants for a folder

import numpy as np
import argparse
import pandas as pd
from netCDF4 import Dataset
import sys, os
import skimage
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import logging


# Add s4pi module to patch
_S4PI_DIR = os.path.abspath(__file__).split('/')[:-5]
_S4PI_DIR = os.path.join('/',*_S4PI_DIR)
sys.path.append(_S4PI_DIR+'/4piuvsun/')
from s4pi.data.preprocessing import loadAIAMap

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True)
    parser.add_argument('--remove_off_limb', action='store_true', help='Remove Off-limb')
    parser.add_argument('--divide',dest='divide',default=4,required=True, type=int)
    args = parser.parse_args()
    return args

def handleStd(index_aia_i):

    divide = 4
    AIA_sample = np.asarray([np.expand_dims(divide*divide*skimage.transform.downscale_local_mean(loadAIAMap(aia_file).data, (divide, divide)), axis=0) for aia_file in index_aia_i], dtype = np.float64 )
    
    return AIA_sample

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    args = parse_args()
    args.base = "%s/" % (args.base)
    divide = args.divide

    # Load nc file
    LOG.info('Loading EVE_irradiance.nc')
    eve = Dataset(args.base + 'EVE_irradiance.nc', "r", format="NETCDF4")
    train = pd.read_csv(args.base+"/train.csv")
    
    
    LOG.info('Calculate mean and std of EVE lines')
    Y = eve.variables['irradiance'][:]
    line_indices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,14])

    Y = Y[train['eve_indices'].values,:][:,line_indices]
    Y[Y<=0] =  0.0
    
    YMean = np.mean(Y,axis=0)
    YStd = np.std(Y,axis=0)

    np.save("%s/eve_mean.npy" % args.base, np.ma.filled(YMean, np.nan))
    np.save("%s/eve_std.npy" % args.base, np.ma.filled(YStd, np.nan))    

    #sqrt
    YSqrt = np.sqrt(Y)
    YSqrtMean = np.mean(YSqrt,axis=0)
    YSqrtStd = np.std(YSqrt,axis=0)

    np.save("%s/eve_sqrt_mean.npy" % args.base, np.ma.filled(YSqrtMean, np.nan))
    np.save("%s/eve_sqrt_std.npy" % args.base, np.ma.filled(YSqrtStd, np.nan))


    aia_columns = [col for col in train.columns if 'AIA' in col]

    fnList = []
    for index, row in tqdm(train.iterrows()):
        fnList.append(row[aia_columns].tolist())

    LOG.info('Processing AIA files')

    AIA_samples = process_map(handleStd, fnList, chunksize=5)
    AIA_samples = np.concatenate(AIA_samples,axis=1)
    

    total_finite = np.sum(np.isfinite(AIA_samples), axis=(0,2,3))
    valid_entries = total_finite == np.max(total_finite)

    if np.sum(valid_entries) < train.shape[0]:
        LOG.info('Clean problematic files and remove them from the training csv')
        train = train.loc[valid_entries,:].reset_index(drop=True)
        train.to_csv(args.base+"/train.csv", index=False)
        AIA_samples = AIA_samples[:, valid_entries, : , :]

    LOG.info('Calculating mean and std for AIA files')
    AIA_samples_sqrt = np.sqrt(AIA_samples)
    AIA_m_sqrt = np.nanmean(AIA_samples_sqrt,axis=(1,2,3))
    AIA_s_sqrt = np.nanstd(AIA_samples_sqrt,axis=(1,2,3))

    LOG.info('AIA_m_sqrt:', AIA_m_sqrt)
    LOG.info('AIA_s_sqrt:', AIA_s_sqrt)

    np.save("%s/aia_sqrt_mean.npy" % args.base,AIA_m_sqrt)
    np.save("%s/aia_sqrt_std.npy" % args.base,AIA_s_sqrt)

    AIA_m = np.nanmean(AIA_samples,axis=(1,2,3))
    AIA_s = np.nanstd(AIA_samples,axis=(1,2,3))

    LOG.info('AIA_m:', AIA_m)
    LOG.info('AIA_s:', AIA_s)

    np.save("%s/aia_mean.npy" % args.base,AIA_m)
    np.save("%s/aia_std.npy" % args.base,AIA_s)

    eve.close()
    

