#!/usr/bin/env python
#Make normalization constants for a folder

import numpy as np
import argparse
import pandas as pd
import pdb
import json
import sys, os
import skimage

# Add modile to patch
_S4PI_DIR = os.path.abspath(__file__).split('/')[:-3]
_S4PI_DIR = os.path.join('/',*_S4PI_DIR)
sys.path.append(_S4PI_DIR+'/4piuvsun/')
from s4pi.data.preprocessing import loadAIAMap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True)
    parser.add_argument('--remove_off_limb', action='store_true', help='Remove Off-limb')
    parser.add_argument('--divide',dest='divide',default=4,required=True)
    args = parser.parse_args()
    return args


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    args = parse_args()
    args.base = "%s/" % (args.base)
    divide = args.divide

    # Load Json file
    eve = json.load(open(args.base+"/EVE.json"))  #loading dictionary with eve data
    train = pd.read_csv(args.base+"/train.csv")
    
    Y = eve["data"][train['eve_indices'],[0,1,2,3,4,5,6,7,8,9,10,11,12,14]]
    Y[Y<=0] =  0.0
    
    YMean = np.mean(Y,axis=0)
    YStd = np.std(Y,axis=0)

    np.save("%s/eve_mean.npy" % args.base,YMean)
    np.save("%s/eve_std.npy" % args.base,YStd)    

    #sqrt
    YSqrt = np.sqrt(Y)
    YSqrtMean = np.mean(YSqrt,axis=0)
    YSqrtStd = np.std(YSqrt,axis=0)

    np.save("%s/eve_sqrt_mean.npy" % args.base,YSqrtMean)
    np.save("%s/eve_sqrt_std.npy" % args.base,YSqrtStd)

    print(YSqrtMean)
    print(YSqrtStd)

    aia_columns = [col for col in train.columns if 'AIA' in col]

    AIA_samples = []
    AIA_samples_sqrt = []

    for index, row in train.iterrows():
        X = np.asarray( [np.expand_dims(divide*divide*skimage.transform.downscale_local_mean(loadAIAMap(row[channel], remove_off_limb=args.remove_off_limb), (divide, divide)), axis=0) for channel in aia_columns], dtype = np.float64 )
        Xsqrt = np.sqrt(X)
        AIA_samples_sqrt.append(Xsqrt)
        AIA_samples.append(X)

    # AIA_samples = []
    # AIA_samples_sqrt = []
    # for index in range(0,Y.shape[0],20):
    #     print("%d/%d" % (index,Y.shape[0]))

    #     X = np.asarray( [np.load(channel.replace(".fits",""))['x'] for channel in index_aia[index, :]] ).astype(dtype=np.float)
    #     Xsqrt = np.sqrt(X)
    #     AIA_samples_sqrt.append(np.expand_dims(Xsqrt,axis=0))
    #     AIA_samples.append(np.expand_dims(X,axis=0))        

    AIA_samples_sqrt = np.concatenate(AIA_samples_sqrt,axis=0)
    AIA_m_sqrt = np.mean(AIA_samples_sqrt,axis=(0,2,3))
    AIA_s_sqrt = np.std(AIA_samples_sqrt,axis=(0,2,3))

    np.save("%s/aia_sqrt_mean.npy" % args.base,AIA_m_sqrt)
    np.save("%s/aia_sqrt_std.npy" % args.base,AIA_s_sqrt)
    
    AIA_samples = np.concatenate(AIA_samples,axis=0)
    AIA_m = np.mean(AIA_samples,axis=(0,2,3))
    AIA_s = np.std(AIA_samples,axis=(0,2,3))

    np.save("%s/aia_mean.npy" % args.base,AIA_m)
    np.save("%s/aia_std.npy" % args.base,AIA_s)
    

