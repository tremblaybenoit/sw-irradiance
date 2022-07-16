import numpy as np
import pandas as pd
import os
import dateutil.parser as dt
from tqdm import tqdm
import glob
from netCDF4 import Dataset
import datetime
import argparse


def generate_time_matches(aia_path,eve_path,output_path, wavelengths, cutoff_eve = 10, cutoff_aia = 10*60, debug=True):
    """_summary_

    Args:
        aia_path (str): path to directory of aia files
        eve_path (str): path to cdf file
        output_path (str): path to store output csv file
        wavelengths: list of wavelengths that are needed
        cutoff_eve(int): cutoff for time delta (difference between AIA and EVE file in time) in seconds
        cutoff_aia(int): cutoff for time delta (difference between AIA images in different wavelengths) in seconds
        debug (bool): Only do 10 AIA matches 

    Returns:
        csv: csv with aia filenames, aia iso dates, eve iso dates, eve indices, and time deltas
    """

    nb_wavelengths = len(wavelengths)

    # List of filenames, per wavelength
    aia_filenames = [[f for f in sorted(glob.glob(aia_path+'/%s/aia%s_*.fits' % (wl, wl)))] for wl in wavelengths]

    eve = Dataset(eve_path, "r", format="NETCDF4")

    pbar_convert_dates = tqdm(eve.variables['isoDate'][:])
    eve_dates = [dt.isoparse(i) for i in pbar_convert_dates]

    aia_dates = [[name.split("_")[-1].split('.')[0]+'z' for name in aia_filenames[i]] for i in range(nb_wavelengths)]

    if debug:
        aia_iso_dates = [[dt.isoparse(date) for date in aia_dates[i][0:10]] for i in range(nb_wavelengths)]
    else:
        aia_iso_dates = [[dt.isoparse(date) for date in aia_dates[i]] for i in range(nb_wavelengths)]

    #creating empty repos to store results of match and times
    eve_res = []
    eve_times = []
    aia_res = [[] for i in range(nb_wavelengths)]
    threshold_eve = datetime.timedelta(seconds=cutoff_eve) #cutoff time for time match
    threshold_aia = datetime.timedelta(seconds=cutoff_aia)
    pbar_finding_matches = tqdm(aia_iso_dates[0])

    #looping through AIA filenames to find matching EVE files
    for aia_date in pbar_finding_matches:
        eve_ans = min(eve_dates, key=lambda sub: abs(sub - aia_date))
        aia_ans = [min(aia_iso_dates[i], key=lambda sub: abs(sub - aia_date)) for i in range(nb_wavelengths)]
        if abs(eve_ans - aia_date) <= threshold_eve and np.amax([abs(aia_ans[i] - aia_date) for i in range(nb_wavelengths)]) <= threshold_aia:
            eve_times += abs(eve_ans - aia_date),
            eve_res+= eve_ans,
            for i in range(nb_wavelengths):
                aia_res[i]+=[aia_ans[i]]
        pbar_finding_matches.set_description("Processing %s" % aia_date)

    eve_idx = [eve_dates.index(date) for date in eve_res] #storing indices in EVE json file
    aia_idx = [[(aia_iso_dates[i]).index(date) for date in aia_res[i]] for i in range(nb_wavelengths)]
    aia_selections = [[aia_filenames[j][i] for i in aia_idx[j]] for j in range(nb_wavelengths)]
    
    match = pd.DataFrame({"eve_indices": eve_idx, 
                          "eve_dates":eve_res,
                          "time_delta":eve_times})
    for i in range(nb_wavelengths):
        match.insert(i+2, 'AIA'+wavelengths[i], aia_selections[i], True)

    # Save
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    filename = output_path+'/matches_eve_aia'
    for i in wavelengths:
        filename = filename+'_'+i
    match.to_csv(filename+'.csv', index=False)

    eve.close()

if __name__ == "__main__":

    # Commands 
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-eve_path', type=str, default="/home/miraflorista/sw-irradiance/data/EVE/EVE.json",
                   help='eve_path')
    p.add_argument('-aia_path', type=str, default="/mnt/aia-jsoc",
                   help='aia_path')
    p.add_argument('-out_path', type=str, default="/home/benoit_tremblay_23",
                   help='out_path')
    p.add_argument('-wavelengths', type=str, default=['171', '193', '211', '304'],
                   help='wavelengths')
    p.add_argument('-cutoff_eve', type=float, default=600,
                   help='cutoff_eve')
    p.add_argument('-cutoff_aia', type=float, default=600,
                   help='cutoff_aia')
    p.add_argument('-debug', dest='debug', type=bool, default=False, help='Only process a few files')
    args = p.parse_args()

    eve_path = args.eve_path
    aia_path = args.aia_path
    out_path = args.out_path
    wavelengths = args.wavelengths
    cutoff_eve = args.cutoff_eve
    cutoff_aia = args.cutoff_aia
    debug_flag = args.debug

    matches = generate_time_matches(aia_path,eve_path, out_path, wavelengths, cutoff_eve=cutoff_eve, cutoff_aia=cutoff_aia, debug=debug_flag)


