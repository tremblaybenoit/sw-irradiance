#!/usr/bin/env python3

'''
Pytorch Dataset class to load AIA and EVE dataset.
'''


import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import os, sys
import skimage.transform
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
                 index_folder, 
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

        """_summary_

        Args:
            EVE_path: str
                File containing the EVE residuals
            index_folder: str 
                Path to train.csv, etc.
            resolution: int
                resolution of the image used to train the NN (i.e. 256)
            EVE_scale: ??
                ??
            Eve_sigmoid: ??
                ??
            
        Params:
            split: str
                which split is being used (i.e. train)
            remove_off_limb: bool
                whether to remove the corona during prep
            AIA_transform: ??
                ??
            flip: bool
                Whether to augment by flipping
            zscore: bool
                Whether to apply scores to ????
            self_mean_normalize: bool
                Whether to normalize each image to itself
            debug: bool
                Only load a little bit of each csv            

        Returns:
            : 
        """
        
        ''' Input path for aia and eve index files, as well as data path for EVE.
            We load EVE during init, but load AIA images on the fly'''

        ### data split (train, val, test)
        self.split = split  
        
        ### do we perform random flips?
        self.flip = flip
        
        # Remove offlimb during dataload
        self.remove_off_limb = remove_off_limb

        ### load indices from csv file for the given split
        # Benito: index_folder is a path
        df_indices = pd.read_csv(index_folder+'/'+split+'.csv')
        if debug:
            df_indices = df_indices.loc[0:10,:]
        
        ### resolution.
        self.resolution = resolution
        
        self.self_mean_normalize = self_mean_normalize
        
        ### all AIA channels. first two columns are junk
        # Benito: Is that the case for us?
        aia_columns = [col for col in df_indices.columns if 'AIA' in col]
        self.index_aia = df_indices[aia_columns].values.tolist()

        ### last column is EVE index
        self.index_eve = np.asarray(df_indices["eve_indices"]).astype(int)
       
        ### EVE processing. What scaling do we use? Do we apply sigmoid? Pass means and stds (computed on scaled EVE)
        ### They are located in the index file (csv) path)
        ### Name is hardcoded based on David's normalization files
        
        self.EVE_scale = EVE_scale
        self.EVE_sigmoid = EVE_sigmoid
        self.zscore = zscore
        
        self.EVE_means = np.load(index_folder + '/eve_'+EVE_scale+'_mean.npy')
        self.EVE_stds = np.load(index_folder + '/eve_'+EVE_scale+'_std.npy')
        if (EVE_sigmoid):
            self.EVE_means = np.load(index_folder + '/eve_'+EVE_scale+'sigmoid'+'_mean.npy')
            self.EVE_stds = np.load(index_folder + '/eve_'+EVE_scale+'sigmoid'+'_std.npy')
        
        if (not zscore):
            self.EVE_means = np.load(index_folder + '/eve_mean.npy')
            self.EVE_stds = np.load(index_folder + '/eve_std.npy')

        self.line_indices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,14])
        eve = nc.Dataset(EVE_path, "r", format="NETCDF4")
        full_EVE = eve.variables['irradiance'][:][:, self.line_indices]
        self.EVE = full_EVE[self.index_eve,:]
        eve.close()

        predictionDB = nc.Dataset( index_folder + 'EVE_linear_pred_' + split + '.nc')
        self.EVE_residual = predictionDB.variables['irradiance'][:] - predictionDB.variables['pred_irradiance'][:]
        predictionDB.close()

        ### AIA transform : means and stds of sqrt(AIA)
        self.AIA_transform = AIA_transform

        ### Check for inconsistencies
        #data length
        if (len(self.index_eve) != len(self.index_aia)):
            raise ValueError('Time length of EVE and AIA are different')
        
        print('Loaded ' + split + ' Dataset with ' + str(len(self.index_eve))+' examples' )



    def __len__(self):

        ### return the number of time steps
        return len(self.index_eve)

    def __getitem__(self, index):

        AIA_down = loadAIAStack(self.index_aia[index], resolution=self.resolution, remove_off_limb=self.remove_off_limb, off_limb_val=0, remove_nans=True)
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
            
       
        EVE_sample = torch.from_numpy( eve_scale(self.EVE_residual[index, :], self.EVE_means, 
                                      self.EVE_stds, self.EVE_scale, self.EVE_sigmoid, self.zscore).astype( np.float32) ).float()
    
        if (self.AIA_transform):
            AIA_sample = self.AIA_transform(AIA_sample)

        return AIA_sample, EVE_sample
