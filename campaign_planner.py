# -*- coding: utf-8 -*-
"""
Planner for field campaign
"""

import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
from matplotlib.path import Path
import glob
import utm
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
from matplotlib.markers import MarkerStyle
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Functions
def three_point_star():
    # Points of a 3-pointed star (scaled and centered)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points (3 outer, 3 inner)
    outer_radius = 1
    inner_radius = 0.1
    coords = []

    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        coords.append((x, y))

    coords.append(coords[0])  # close the shape
    return Path(coords)

def angle_difference_deg(a1, a2):
    return (a2 - a1 + 180) % 360 - 180 

def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))

def dual_Doppler(x1,x2,y1,y2):
    
    x=np.arange(-1000,1001,10)#[m] dual-Doppler x
    y=np.arange(-1000,1001,10)#[m] dual-Doppler y 
    
    DD=xr.Dataset()
    DD['x']=xr.DataArray(data=x+x1,coords={'x':x+x1})
    DD['y']=xr.DataArray(data=y+y1,coords={'y':y+y1})

    #define angles
    DD['chi1']=np.degrees(np.arctan2(DD.y-y1,DD.x-x1))
    DD['chi2']=np.degrees(np.arctan2(DD.y-y2,DD.x-x2))
    DD['dchi']=angle_difference_deg(DD['chi1'],DD['chi2'])
    DD['chi_avg']=np.degrees(np.arctan2(sind(DD.chi1)+sind(DD.chi2),cosd(DD.chi1)+cosd(DD.chi2)))
    DD['alpha_u']=angle_difference_deg(DD['chi_avg'],0)
    DD['alpha_v']=angle_difference_deg(DD['chi_avg'],90)

    #uncertainties
    Nu=(sind(DD['alpha_u']+DD['dchi']/2))**2+(sind(DD['alpha_u']-DD['dchi']/2))**2
    Nv=(sind(DD['alpha_v']+DD['dchi']/2))**2+(sind(DD['alpha_v']-DD['dchi']/2))**2
    D=sind(DD['dchi'])**2

    DD['sigma_u']=(Nu/(D+10**-10))**0.5
    DD['sigma_v']=(Nv/(D+10**-10))**0.5
        
    return DD

#%% Inputs
source_topo=os.path.join(cd,'data/FC_topo_v2.nc')#source of terrain data
source_nwtc=os.path.join(cd,'data/FC_assets.xlsx')#source nwtc sites
source_sens=os.path.join(cd,'data/FC_sensing.xlsx')#source remote sensing fleet
source_m2=os.path.join(cd,'data/nwtc.m2.b0/*nc')#source of M2 wind data

bins_ws=np.array([0,2.5,5,7.5,10,15,25])#[m/s] bins in wind speed
bins_wd=np.arange(0,360,30)#[deg] bins in wind direction

#graphics
markers={'Research pad':'s','Wind turbine':MarkerStyle(three_point_star()),'Wind sensing':'*','Met tower':'o',
         'SL':'^','TP':'^','CEIL':'^','PL':'^'}
marker_sizes={'Research pad':5,'Wind turbine':25,'Wind sensing':15,'Met tower':15,
              'SL':20,'TP':15,'CEIL':5,'PL':10}
marker_colors={'SL':'r','TP':'orange','CEIL':'c','PL':'w'}

#%% Initialization

#read data
Topo=xr.open_dataset(source_topo)
FC=pd.read_excel(source_nwtc).set_index('Site')
Sens=pd.read_excel(source_sens).set_index('Instrument ID')
M2=xr.open_mfdataset(glob.glob(source_m2))

#locations
xy=utm.from_latlon(FC['Lat'].values,FC['Lon'].values)
FC['x']=xy[0]
FC['y']=xy[1]
        
#grid and select elevation data
x=Topo.x.values
y=Topo.y.values
Z=Topo.z.values.T

#Cartesianize sites
FC['x_utm'],FC['y_utm'],FC['zone_utm1'],FC['zone_utm2']=utm.from_latlon(FC['Lat'].values, FC['Lon'].values)
f = RegularGridInterpolator((y,x), Z)  
FC['z']=f(np.column_stack((FC['y_utm'], FC['x_utm'])))

#dual-Doppler maps
lidar_DD1=Sens.index[Sens['Use']=='dual-Doppler'][0]
lidar_DD2=Sens.index[Sens['Use']=='dual-Doppler'][1]
x_DD1=FC.loc[Sens.loc[lidar_DD1]['Site']]['x_utm']
x_DD2=FC.loc[Sens.loc[lidar_DD2]['Site']]['x_utm']
y_DD1=FC.loc[Sens.loc[lidar_DD1]['Site']]['y_utm']
y_DD2=FC.loc[Sens.loc[lidar_DD2]['Site']]['y_utm']

#dual-Doppler error
DD=dual_Doppler(x_DD1, x_DD2, y_DD1, y_DD2)

#climatology
M2=M2.where(M2.WS_5m>0)
N=     stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='count',bins=[bins_ws,bins_wd])[0]
ws_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='mean',bins=[bins_ws,bins_wd])[0]
wd_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WD_5m.values,statistic='mean',bins=[bins_ws,bins_wd])[0]

DD['N']=xr.DataArray(N/np.sum(N),coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})
DD['ws_avg']=xr.DataArray(ws_avg,    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})
DD['wd_avg']=xr.DataArray(wd_avg,    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})

DD['sigma_ws']=(((cosd(270-DD['wd_avg']))**2*DD['sigma_u']**2+\
                 (sind(270-DD['wd_avg']))**2*DD['sigma_v']**2)**0.5*DD['N']).sum(dim='ws').sum(dim='wd')
    
DD['sigma_wd']=(((sind(270-DD['wd_avg']))**2*DD['sigma_u']**2+\
                 (cosd(270-DD['wd_avg']))**2*DD['sigma_v']**2)**0.5/DD['ws_avg']*DD['N']).sum(dim='ws').sum(dim='wd')
    
#%% Plots

#map
plt.figure(figsize=(20,15))
plt.pcolor(x,y,Z,cmap='summer',vmin=np.min(FC['z'])-30,vmax=np.max(FC['z'])+10)
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')
plt.colorbar(label='Altitude above sea level [m]')
for s in FC.index:
    plt.plot(FC.loc[s].x_utm,FC.loc[s].y_utm,'.k',
             marker=markers[FC.loc[s].Description],markersize=marker_sizes[FC.loc[s].Description])
for s in Sens.index:
    plt.plot(FC.loc[Sens.loc[s].Site].x_utm,FC.loc[Sens.loc[s].Site].y_utm,'.',
             marker=markers[Sens.loc[s].Type],markersize=marker_sizes[Sens.loc[s].Type],
             color=marker_colors[Sens.loc[s].Type])
    
plt.xlim([np.min(FC['x_utm'])-500,np.max(FC['x_utm'])+500])
plt.ylim([np.min(FC['y_utm'])-500,np.max(FC['y_utm'])+500])

#check dual-Doppler components
plt.figure(figsize=(25,15))
ax=plt.subplot(2,3,1)
cf=plt.contourf(DD.x,DD.y,DD.chi1%360,np.arange(0,360,10),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar(cf,label=r'$\chi_1$ [$^\circ$]')

ax=plt.subplot(2,3,2)
cf=plt.contourf(DD.x,DD.y,DD.chi2%360,np.arange(0,360,10),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar(cf,label=r'$\chi_2$ [$^\circ$]')

ax=plt.subplot(2,3,3)
cf=plt.contourf(DD.x,DD.y,DD.dchi%360,np.arange(0,360,10),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar(cf,label=r'$\Delta \chi$ [$^\circ$]')

ax=plt.subplot(2,3,4)
cf=plt.contourf(DD.x,DD.y,DD.alpha_u%360,np.arange(0,360,10),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar(cf,label=r'$\alpha_u$ [$^\circ$]')

ax=plt.subplot(2,3,5)
cf=plt.contourf(DD.x,DD.y,DD.alpha_v%360,np.arange(0,360,10),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar(cf,label=r'$\alpha_v$ [$^\circ$]')

ax=plt.subplot(2,3,6)
cf=plt.contourf(DD.x,DD.y,DD.chi_avg%360,np.arange(0,360,10),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar(cf,label=r'$\overline{\chi}$ [$^\circ$]')
plt.tight_layout()

#dual-Doppler error (u,v)
plt.figure(figsize=(20,7))
ax=plt.subplot(1,2,1)
cf=plt.contourf(DD.x,DD.y,DD.sigma_u,np.arange(0,11),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
for s in FC.index:
    plt.plot(FC.loc[s].x_utm,FC.loc[s].y_utm,'.k',
             marker=markers[FC.loc[s].Description],markersize=marker_sizes[FC.loc[s].Description])
plt.colorbar(cf,label=r'Error factor of $u$')
   
ax=plt.subplot(1,2,2)
cf=plt.contourf(DD.x,DD.y,DD.sigma_v,np.arange(0,11),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
ax.set_yticklabels([])
plt.xticks(rotation=30) 
for s in FC.index:
    plt.plot(FC.loc[s].x_utm,FC.loc[s].y_utm,'.k',
             marker=markers[FC.loc[s].Description],markersize=marker_sizes[FC.loc[s].Description])
plt.colorbar(cf,label=r'Error factor of $v$')
plt.tight_layout()

#dual-Doppler error (WS,WD)
plt.figure(figsize=(20,7))
ax=plt.subplot(1,2,1)
cf=plt.contourf(DD.x,DD.y,DD.sigma_ws,np.arange(0,11),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
for s in FC.index:
    plt.plot(FC.loc[s].x_utm,FC.loc[s].y_utm,'.k',
             marker=markers[FC.loc[s].Description],markersize=marker_sizes[FC.loc[s].Description])
plt.colorbar(cf,label=r'Error factor of wind speed')
ax=plt.subplot(1,2,2)
cf=plt.contourf(DD.x,DD.y,DD.sigma_wd,np.arange(0,11),cmap='RdYlGn_r',extend='both')
ax.set_aspect('equal')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xticks(rotation=30) 
ax.set_yticklabels([])
for s in FC.index:
    plt.plot(FC.loc[s].x_utm,FC.loc[s].y_utm,'.k',
             marker=markers[FC.loc[s].Description],markersize=marker_sizes[FC.loc[s].Description])
plt.colorbar(cf,label=r'Error factor wind direction [$^\circ$/m s$^{-1}$]')
plt.tight_layout()
    
#climatology
plt.figure(figsize=(20,5))
ax=plt.subplot(1,3,1)
pc=plt.pcolor(DD.ws,DD.wd,DD.N.T,cmap='plasma',vmin=0,vmax=0.1)
plt.xlabel('Wind speed [m s$^{-1}$]')
plt.ylabel('Wind direction [$^\circ$]')
plt.colorbar(pc,label='Probability')
ax=plt.subplot(1,3,2)
pc=plt.pcolor(DD.ws,DD.wd,DD.ws_avg.T,cmap='plasma',vmin=0,vmax=20)
plt.xlabel('Wind speed [m s$^{-1}$]')
plt.ylabel('Wind direction [$^\circ$]')
plt.colorbar(pc,label='Mean wind speed [m s$^{-1}$')
ax=plt.subplot(1,3,3)
pc=plt.pcolor(DD.ws,DD.wd,DD.wd_avg.T,cmap='plasma',vmin=0,vmax=360)
plt.xlabel('Wind speed [m s$^{-1}$]')
plt.ylabel('Wind direction [$^\circ$]')
plt.colorbar(pc,label='Mean wind direction [$^\circ$]')
plt.tight_layout()

# plt.figure(figsize=(20,15))
# for i_wd in range(len(DD.wd)):
#     ax=plt.subplot(2,2,i_wd+1)
#     cf=plt.contourf(DD.x,DD.y,DD.sigma_ws.isel(wd=i_wd),np.arange(0,11),cmap='RdYlGn_r',extend='both')
#     ax.arrow(x_DD1+800,y_DD1+800,np.cos(np.radians(270-wd[i_wd]))*200,np.sin(np.radians(270-wd[i_wd]))*200,head_width=300, head_length=100, fc='b', ec='k',width=200,alpha=0.5)
#     ax.set_aspect('equal')
#     plt.xlabel('$x$ [m]')
#     plt.ylabel('$y$ [m]')
#     plt.colorbar(cf,label='Error factor wind speed')
#     for s in FC.index:
#         plt.plot(FC.loc[s].x_utm,FC.loc[s].y_utm,'.k',marker=markers[FC.loc[s].Description],markersize=5)
        
# plt.figure(figsize=(20,15))
# for i_wd in range(len(DD.wd)):
#     ax=plt.subplot(2,2,i_wd+1)
#     cf=plt.contourf(DD.x,DD.y,DD.sigma_wd.isel(wd=i_wd),np.arange(0,11),cmap='RdYlGn_r',extend='both')
#     ax.arrow(x_DD1+800,y_DD1+800,np.cos(np.radians(270-wd[i_wd]))*200,np.sin(np.radians(270-wd[i_wd]))*200,head_width=300, head_length=100, fc='b', ec='k',width=200,alpha=0.5)
#     ax.set_aspect('equal')
#     plt.xlabel('$x$ [m]')
#     plt.ylabel('$y$ [m]')
#     plt.colorbar(cf,label='Error factor wind direction')
#     for s in FC.index:
#         plt.plot(FC.loc[s].x_utm,FC.loc[s].y_utm,'.k',marker=markers[FC.loc[s].Description],markersize=5)
