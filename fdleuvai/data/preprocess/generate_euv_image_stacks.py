
import numpy as np
import argparse
import pandas as pd
from netCDF4 import Dataset
import sys, os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import logging
from os.path import exists

# Add utils module to load stacks
_FDLEUVAI_DIR = os.path.abspath(__file__).split('/')[:-4]
_FDLEUVAI_DIR = os.path.join('/',*_FDLEUVAI_DIR)
sys.path.append(_FDLEUVAI_DIR)
from fdleuvai.data.utils import loadAIAStack, str2bool

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


def handleStd(index_aia_i):

    # Extract filename from index_aia_i (remove aia_path)
    filename = (index_aia_i[0].replace(aia_path, '')).split('_')[1]
    # Replace .fits by .npy
    filename = filename.replace('.fits', '.npy')

    if exists(stack_outpath+'/aia_'+filename):
        LOG.info(f'{filename} exists.')
    else:
        aia_stack = loadAIAStack(index_aia_i, resolution=resolution, remove_off_limb=remove_off_limb, off_limb_val=0, remove_nans=True)
        # Save stack
        np.save(stack_outpath+'/aia_'+filename, aia_stack)

    # LOG.info(stack_outpath+'/aia_'+filename)

    return stack_outpath+'/aia_'+filename

def parse_args():
    # Benito: Should we also extract the relevant EVE data?
    # Commands 
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument('-eve_path', type=str, default="/home/miraflorista/sw-irradiance/data/EVE/EVE.json",
    #            help='eve_path')
    p.add_argument('-aia_path', dest='aia_path', type=str, default="/mnt/miniset/aia-stacks",
                   help='aia_path')
    p.add_argument('-matches', dest='matches', type=str, default="/home/andres_munoz_j/sw-irr-output/matches_eve_aia_171_193_211_304.csv",
                   help='matches')
    p.add_argument('-stack_outpath', dest='stack_outpath', type=str, default="/mnt/miniset/aia-stacks",
                   help='out_path')
    p.add_argument('-debug', dest='debug', type=str2bool, default=False, help='Only process a few files')
    p.add_argument('-resolution', dest='resolution', type=int, default=256, help='Resolution of the output images')
    p.add_argument('-remove_off_limb', dest='remove_off_limb', type=str2bool, default=False, help='Remove Off-limb')
    args = p.parse_args()
    return args 

if __name__ == "__main__":

    # Partser
    args = parse_args()
    #eve_path = args.eve_path
    global aia_path
    aia_path = args.aia_path
    global stack_outpath
    stack_outpath = args.stack_outpath
    matches_file = args.matches
    global resolution
    resolution = args.resolution
    global remove_off_limb
    remove_off_limb = args.remove_off_limb
    debug = args.debug

    # Load indices
    matches = pd.read_csv(matches_file)
    if debug:
        matches = matches.loc[0:10, :]

    # Benito: We could technically save this as a npy file and be done with it.
    # y = eve_data[matches['eve_indices'].values,:]

    # Extract filenames for stacks
    Xs = []
    fnList = []
    aia_columns = [col for col in matches.columns if 'AIA' in col]
    for index, row in tqdm(matches.iterrows()):
        fnList.append(row[aia_columns].tolist())

    # Path for output
    if not os.path.exists(stack_outpath):
        os.makedirs(stack_outpath, exist_ok=True)

    # Stacks
    # Benito: Since we pass filenames as input, can't we take advantage of this when saving?
    #         Well, parallization might make it problematic
    AIA_samples = process_map(handleStd, fnList, chunksize=5)
    
    # Benito:
    # With all the files created, look for filenames (they are supposed to be in chronological order)
    # aia_filenames = sorted(glob.glob(stack_outpath+'/*npy'))


    # Save
    if debug:
        matches = matches.loc[0:len(AIA_samples),:]

    # Benito: Add filenames to csv file
    #         Or if we make a file for EVE, no need for this.

    matches['aia_stack'] = AIA_samples
    matches.to_csv(matches_file.replace('.csv', '_stacks.csv'), index=False)


    