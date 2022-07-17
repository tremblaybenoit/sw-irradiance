#!/usr/bin/env python3

'''
Pytorch Dataset class to load AIA and EVE dataset.
Currently takes all 9 AIA channels and outputs 14 EVE channels. For now in EVE we discard MEGS-B data because
undersampled, as well as channel 14 because heavily undersampled.
'''


import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import os, sys
import math
import logging
import netCDF4 as nc

# Add utils module to load stacks
_FDLEUVAI_DIR = os.path.abspath(__file__).split('/')[:-3]
_FDLEUVAI_DIR = os.path.join('/',*_FDLEUVAI_DIR)
sys.path.append(_FDLEUVAI_DIR)
from fdleuvai.data.utils import loadAIAStack


# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


def aia_scale(aia_sample, zscore = True, self_mean_normalize=False):
    if not self_mean_normalize:
        bad = np.where(aia_sample <= 0.0)
        aia_sample[bad] = 0.0
    
    if (zscore): ### if zscore return sqrt 
        return np.sign(aia_sample) * np.sqrt(np.abs(aia_sample))
    else: ### otherwise we just wanna divide the unsrt image by the means
        return aia_sample
      
    

### means and stds for EVE preprocessing, computed on sqrt of EVE


### def sigmoid for normalization below
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


### scaling for mean and std processing
def eve_scale(eve_sample, eve_means, eve_stds, scale = 'sqrt', eve_sigmoid = False, zscore = True):   
    
    #### remove -1 values
    bad_eve = np.where(eve_sample ==-1)
    eve_sample[bad_eve] = 0.0    
    if (zscore):
        if (scale == 'sqrt'): 

            eve_sample = np.sqrt(eve_sample)       
            if (eve_sigmoid):
                eve_sample = sigmoid(eve_sample)            
            eve_sample -= eve_means
            eve_sample /= eve_stds

        elif (scale == 'log'):

            eve_sample = np.log1p(eve_sample)       
            if (eve_sigmoid):
                eve_sample = sigmoid(eve_sample)            
            eve_sample -= eve_means
            eve_sample /= eve_stds 
            
        else :
            raise ValueError('Unknown scaling argument')
        
    else: ### don't do any scaling just divide by the means 
        eve_sample /= eve_means
        
    return eve_sample



class SW_Dataset(Dataset):

    ''' Dataset class to get inputs and labels for AIA-->EVE mapping'''

    def __init__(self, 
                EVE_path, 
                AIA_root, 
                index_file, 
                resolution, 
                EVE_scale, 
                EVE_sigmoid, 
                split = 'train',
                remove_off_limb=False, 
                AIA_transform = None, 
                flip = False, 
                zscore = True, 
                self_mean_normalize=False,
                debug = False):

        ''' Input path for aia and eve index files, as well as data path for EVE.
            We load EVE during init, but load AIA images on the fly'''

        ### data split (train, val, test)
        self.split = split  
        
        ### do we perform random flips?
        self.flip = flip

        # Remove offlimb during dataload
        self.remove_off_limb = remove_off_limb
        
        df_indices = pd.read_csv(index_folder+'/'+split+'.csv')
        if debug:
            df_indices = df_indices.loc[0:10,:]
        
        ### resolution. normal is 224, if 256 we perform crops
        self.resolution = resolution
        
        self.self_mean_normalize = self_mean_normalize
        

        aia_columns = [col for col in df_indices.columns if 'AIA' in col]

        if 'aia_stack' in df_indices.columns:
            self.index_aia = df_indices['aia_stack'].values.tolist()
        else:
            self.index_aia = df_indices[aia_columns].values.tolist()

        ### AIA transform : means and stds of sqrt(AIA)
        self.AIA_transform = AIA_transform

            
        #crop arguments
        if (self.crop and self.crop_res == None):
            raise ValueError('If crops are on, please specify a crop resolution')
        if (self.crop and self.crop_res > self.resolution):
            raise ValueError('Cropping resolution must be smaller than initial resolution')
        
        print('Loaded inference dataset with ' + str(len(self.index_aia))+' examples' )

    def __len__(self):

        ### return the number of time steps
        return len(self.index_aia)

    def __getitem__(self, index):

        ### Training in paper is done on 256 images but new data is 512 so we downsample here.
        if type(self.index_aia[index])==list:
            AIA_down = loadAIAStack(self.index_aia[index], resolution=self.resolution, remove_off_limb=self.remove_off_limb, off_limb_val=0, remove_nans=True)
        else:
            AIA_down = np.load(self.index_aia[index])

        AIA_sample = np.concatenate(AIA_down, axis = 0)

        if self.self_mean_normalize:
            AIA_sample = AIA_sample - np.mean(AIA_sample,axis=(1,2),keepdims=True)

        ### random flips
        if (self.flip):
            AIA_sample_temp = AIA_sample
            p = np.random.rand()
            if (p>0.5):
                AIA_sample_temp = np.flip(AIA_sample_temp, axis = 2)
            d = np.random.rand()
            if (d>0.5):
                AIA_sample_temp = np.flip(AIA_sample_temp, axis = 1)
                
            ### need to make a copy because pytorch doesnt support negative strides yet    
            AIA_sample = torch.from_numpy(aia_scale(AIA_sample_temp.copy(), self.zscore,self.self_mean_normalize))
        else:
            AIA_sample = torch.from_numpy(aia_scale(AIA_sample, self.zscore, self.self_mean_normalize))  
            
       
    
        if (self.AIA_transform):
            AIA_sample = self.AIA_transform(AIA_sample)

        return AIA_sample
