import numpy as np
import pandas as pd
import os
import dateutil.parser as dt
from tqdm import tqdm
import glob
import json
import datetime

def generate_time_matches(aia_path,eve_path,output_path,cutoff_seconds = 10*60):
    """_summary_

    Args:
        aia_path (str): path to directory of aia files
        eve_path (str): path to json file
        output_path (str): path to store output csv file
        cutoff_seconds(int): cutoff for time delta (difference between AIA and EVE file in time) in seconds

    Returns:
        csv: csv with aia filenames, aia iso dates, eve iso dates, eve indices, and time deltas
    """

    aia_filenames = os.listdir(aia_path) #generate list of filenames
    eve = json.load(open(eve_path)) #loading dictionary with eve data
    pbar_convert_dates = tqdm(eve["metadata"]["raw_dates"])
    eve["metadata"]["iso_dates"] = [dt.isoparse(i) for i in pbar_convert_dates]

    aia_dates = ["".join(name.split("_")[3:8])+"z" for name in aia_filenames] #recovering date from aia filename
    aia_wavelengths = ["".join(name.split("_")[2]) for name in aia_filenames] #recovering wavelegnthfrom aia filename

    aia_dates
    dt.isoparse(aia_dates[0])
    aia_iso_dates= [dt.isoparse(date) for date in aia_dates]

    #creating empty repos to store results of match and times
    res = []
    times = []
    cutoff = datetime.timedelta(seconds=cutoff_seconds) #cutoff time for time match
    pbar_finding_matches = tqdm(aia_iso_dates)

    #looping through AIA filenames to find matching EVE files
    for aia_date in pbar_finding_matches:
        ans=min(eve["metadata"]["iso_dates"], key=lambda sub: abs(sub - aia_date))  
        if abs(ans - aia_date) <= cutoff:
            times += abs(ans - aia_date),
            res+= ans,
        pbar_finding_matches.set_description("Processing %s" % aia_date)

    match_idx = [eve["metadata"]["iso_dates"].index(date)for date in res] #storing indices in EVE json file

    match = pd.DataFrame({"aia_filenames": aia_filenames , #storing everything in a pandas dataframe
                            "aia_iso_dates": aia_iso_dates,
                            "aia_wavelengths": aia_wavelengths,
                            "eve_indices": match_idx, "eve_dates":res,
                             "time_delta":times})
    
    match.to_csv(output_path+'matches_aia_eve.csv', index=False)

    return match

if __name__ == "__main__":
    eve_path = "/home/miraflorista/sw-irradiance/data/EVE/EVE.json"
    aia_path = "/mnt/miniset/aia"
    output_path = "/home/miraflorista/sw-irradiance/data/"
    matches = generate_time_matches(aia_path,eve_path, output_path,cutoff_seconds= 10*60 )


