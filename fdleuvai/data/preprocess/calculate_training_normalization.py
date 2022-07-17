#!/usr/bin/env python
#Make normalization constants for a folder

import numpy as np
import argparse
import pandas as pd
from netCDF4 import Dataset
import sys, os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import logging


# Add utils module to load stacks
_FDLEUVAI_DIR = os.path.abspath(__file__).split('/')[:-4]
_FDLEUVAI_DIR = os.path.join('/',*_FDLEUVAI_DIR)
sys.path.append(_FDLEUVAI_DIR)
from fdleuvai.data.utils import loadAIAStack

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-base',dest='base',required=True)
    parser.add_argument('-remove_off_limb', dest='remove_off_limb', type=bool, default=False, help='Remove Off-limb')
    parser.add_argument('-debug', dest='debug', type=bool, default=False, help='Only process a few files')
    parser.add_argument('-resolution', dest='resolution', default=256, type=int)
    args = parser.parse_args()
    return args

def handleStd(index_aia_i):
    return loadAIAStack(index_aia_i, resolution=resolution, remove_off_limb=remove_off_limb)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    args = parse_args()
    args.base = "%s/" % (args.base)
    global resolution
    resolution = args.resolution
    global remove_off_limb
    remove_off_limb = args.remove_off_limb

    # Load nc file
    LOG.info('Loading EVE_irradiance.nc')
    eve = Dataset(args.base + 'EVE_irradiance.nc', "r", format="NETCDF4")
    train = pd.read_csv(args.base+"/train.csv")

    if args.debug:
        train = train.loc[0:4,:]    
    
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
    AIA_samples[AIA_samples<0] = 0
    AIA_samples_sqrt = np.sqrt(AIA_samples)
    AIA_m_sqrt = np.nanmean(AIA_samples_sqrt,axis=(1,2,3))
    AIA_s_sqrt = np.nanstd(AIA_samples_sqrt,axis=(1,2,3))

    LOG.info('AIA_m_sqrt')
    print(AIA_m_sqrt)
    LOG.info('AIA_s_sqrt')
    print(AIA_s_sqrt)

    np.save("%s/aia_sqrt_mean.npy" % args.base,AIA_m_sqrt)
    np.save("%s/aia_sqrt_std.npy" % args.base,AIA_s_sqrt)

    AIA_m = np.nanmean(AIA_samples,axis=(1,2,3))
    AIA_s = np.nanstd(AIA_samples,axis=(1,2,3))

    LOG.info('AIA_m')
    print(AIA_m)
    LOG.info('AIA_s')
    print(AIA_s)

    np.save("%s/aia_mean.npy" % args.base,AIA_m)
    np.save("%s/aia_std.npy" % args.base,AIA_s)

    eve.close()
    

