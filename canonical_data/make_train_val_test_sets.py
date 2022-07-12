import numpy as np
from sunpy.map import Map, make_fitswcs_header
import glob
import os, sys
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.linear_model import SGDRegressor
import pdb
import argparse

_S4PI_DIR = os.path.abspath(__file__).split('/')[:-3]
_S4PI_DIR = os.path.join('/',*_S4PI_DIR)
print(_S4PI_DIR)
sys.path.append(_S4PI_DIR)
from s4pi.data.preprocessing import loadAIAMap


def handleStd(index_aia_i, divide=2):
    """Function to downsample (spatially) images.

    Args:
        img: Full-resolution image.

    Returns:
        x: Downsampled image
    """
    # Read all channels
    # AIA_sample = np.asarray( [np.expand_dims(channel, axis=0) for channel in index_aia_i], dtype = np.float64 )
    AIA_sample = np.asarray( [np.expand_dims(read_calibrate_aia(channel), axis=0) for channel in index_aia_i], dtype = np.float64 )
    print(AIA_sample.shape)
    # Factor 4 because new different image resolution (512) than when training (256)
    AIA_sample = np.concatenate(AIA_sample,axis=0)
    print(AIA_sample.shape)
    AIA_down = np.asarray(([np.expand_dims(divide*divide*skimage.transform.downscale_local_mean(AIA_sample[i,:,:], (divide, divide)), axis=0) for i in range(AIA_sample.shape[0])]), dtype=np.float64 )
    AIA_sample = np.concatenate(AIA_down, axis = 0)
    print(AIA_sample.shape)
    # Compute Mean & standard deviation
    X = np.mean(AIA_sample, axis=(1,2))
    X = np.concatenate([X, np.std(AIA_sample, axis=(1,2))], axis=0)
    
    # Return downsampled maps and statistical properties 
    return AIA_sample, np.expand_dims(X, axis=0)

def read_calibrate_aia(filename):
    """Function to read and calibrate SDO/AIA .fits files.

    Args:
        img: Full-resolution image.

    Returns:
        x: Data from AIA
    """
    # Read AIA maps and calibrate using ITI
    aia_map = loadAIAMap(filename)

    # Return only the data
    return aia_map.data

# Main
if __name__ == '__main__':
    
    # Generate training, validation and test sets.
    a = 1

    # Path to data:
    aia_path = '/mnt/miniset/aia'

    # List of filenames, per wavelength
    aia_filenames = [[f for f in glob.glob(aia_path+'/aia_lev1_%sa_*.fits' % (wl))] for
                    wl in ['171', '193', '211', '304']]
    
    # Load Pandas or CSV file

    # Steps
    aia_index = [f for f in aia_filenames[0][]]

    # Preprocessing
    #for i in range(nb_files):
    #    # Read & calibrate the SDO/AIA image
    #    aia_data = read_calibrate_aia(filename)
    #    # 





