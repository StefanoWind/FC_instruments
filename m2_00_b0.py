'''
Format M2 data
'''

import os
cd=os.path.dirname(__file__)
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings
plt.close('all')
warnings.filterwarnings("ignore")

#%% Inputs
source=os.path.join(cd,'data/20240101_M2.csv')
storage=os.path.join(cd,'data/nwtc.m2.b0')

replace=True
column_names = ["DATE", "MST", "Avg Wind Speed @ 5m [m/s]", "Avg Wind Direction @ 5m [deg]"]
time_offset=7#[h]

#%% INitialization
data = pd.read_csv(source, names=column_names, header=0, parse_dates=[["DATE", "MST"]])
os.makedirs(storage,exist_ok=True)

#%% Main
data["DATE_MST"]=data["DATE_MST"]+np.timedelta64(time_offset, 'h')
data["Date"] = data["DATE_MST"].dt.date

# Group by day and store in a dictionary
daily_data = {date: group for date, group in data.groupby("Date")}

for d in daily_data:
    time=daily_data[d]["DATE_MST"].to_numpy()
    filename=os.path.normpath(storage).split(os.sep)[-1]+'.'+str(time[0])[:-10].replace('-','').replace('T','.').replace(':','')+'.nc'
    if not os.path.isfile(filename) or replace==True:
        Output=xr.Dataset()
    
        Output['WS_5m']=xr.DataArray(daily_data[d]["Avg Wind Speed @ 5m [m/s]"].values,coords={'time':time})
        Output['WD_5m']=xr.DataArray(daily_data[d]["Avg Wind Direction @ 5m [deg]"].values,coords={'time':time})

        Output_avg=Output.resample(time="10min").mean()
        Output_avg['WD_5m']=np.degrees(np.arctan2(np.sin(np.radians(Output_avg['WD_5m'])),
                                                  np.cos(np.radians(Output_avg['WD_5m']))))%360
        Output_avg.to_netcdf(os.path.join(storage,filename))
        
    


            
        