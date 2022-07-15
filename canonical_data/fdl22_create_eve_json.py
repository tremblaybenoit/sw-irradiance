
import numpy as np
import dateutil.parser as dt
import json
from tqdm import tqdm
import argparse



def create_eve_json(eve_raw_path,json_outpath):

  eve_date = np.load(eve_raw_path + "/iso.npy",allow_pickle=True)
  eve_irr = np.load(eve_raw_path + "/irradiance.npy",allow_pickle=True)
  eve_jd = np.load(eve_raw_path + "/jd.npy",allow_pickle=True)
  eve_logt = np.load(eve_raw_path + "/logt.npy",allow_pickle=True)
  eve_name = np.load(eve_raw_path + "/name.npy",allow_pickle=True)
  eve_wl = np.load(eve_raw_path + "/wavelength.npy",allow_pickle=True)

  eve= {
    "metadata":{
      "raw_dates":eve_date.tolist(), #iso format dates
      "jd":eve_jd.tolist(), #julian dates
      "logt":eve_logt.tolist(), #temperature
      "name":eve_name.tolist(), #name of channel
      "wavelength":eve_wl.tolist() #wavelength
    },
    "data":eve_irr.tolist()
  }

  with open(json_outpath, "w") as outfile:
      json.dump(eve, outfile)

if __name__ == "__main__":
  p = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('-eve_raw_path', type=str, default="/home/miraflorista/sw-irradiance/data/EVE/raw/",
                  help='eve_raw_path')
  p.add_argument('-json_outpath', type=str, default="/home/miraflorista/sw-irradiance/data/EVE/EVE.json",
                  help='json_outpath')
  args = p.parse_args()

  eve_raw_path = args.eve_raw_path
  json_outpath = args.json_outpath
  create_eve_json(eve_raw_path,json_outpath)