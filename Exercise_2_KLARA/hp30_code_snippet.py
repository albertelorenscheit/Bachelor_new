import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from lib.solar_emphemeris import *
from chaosmagpy.data_utils import timestamp, mjd2000
from hp30client import getKpindex

# Load dark data and store in data frame
dark_data = pd.read_table("ex2_dataset_only_dark.txt", delimiter=",")

# Get Hp30 indices
startdate_whole = timestamp(dark_data['time_stamp'].iloc[0]) #np datetime format
enddate_whole = timestamp(dark_data['time_stamp'].iloc[-1])
startdate = np.datetime64(startdate_whole, 'D') # ensure correct length
enddate = np.datetime64(enddate_whole, 'D')
time_Hp30, Hp30, _ = getKpindex(str(startdate), str(enddate), 'Hp30')

# Convert times to mjd2000
stamp = []
for i in np.arange(0, len(time_Hp30)):
    yr1, mn1, dt1 = time_Hp30[i].split("-")
    hr1, min1, _ = time_Hp30[i].split(":")
    stamp.append(to_mjd2000(int(yr1), int(mn1), int(dt1[:2]), int(hr1[-2:]), int(min1)))

Hp30_indices = pd.DataFrame({'time_Hp30': stamp, 'Hp30': Hp30})