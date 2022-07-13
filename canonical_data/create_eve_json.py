
import numpy as np
import dateutil.parser as dt
import json
from tqdm import tqdm

eve_date = np.load("/home/miraflorista/sw-irradiance/data/EVE/raw/iso.npy",allow_pickle=True)
eve_irr = np.load("/home/miraflorista/sw-irradiance/data/EVE/raw/irradiance.npy",allow_pickle=True)
eve_jd = np.load("/home/miraflorista/sw-irradiance/data/EVE/raw/jd.npy",allow_pickle=True)
eve_logt = np.load("/home/miraflorista/sw-irradiance/data/EVE/raw/logt.npy",allow_pickle=True)
eve_name = np.load("/home/miraflorista/sw-irradiance/data/EVE/raw/name.npy",allow_pickle=True)
eve_wl = np.load("/home/miraflorista/sw-irradiance/data/EVE/raw/wavelength.npy",allow_pickle=True)

eve= {
  "metadata":{
    "raw_dates":eve_date.tolist(),
    "jd":eve_jd.tolist(),
    "logt":eve_logt.tolist(),
    "name":eve_name.tolist(),
    "wavelength":eve_wl.tolist()
  },
  "data":eve_irr.tolist()
}

with open("/home/miraflorista/sw-irradiance/data/EVE/EVE.json", "w") as outfile:
    json.dump(eve, outfile)