# -*- coding: utf-8 -*-
"""
Identify suitable pair of sites for dual-Doppler
"""

import os
cd=os.getcwd()
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib
import pandas as pd
import glob
import utm
from scipy import stats
warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Functions
def angle_difference_deg(a1, a2):
    return (a2 - a1 + 180) % 360 - 180 

def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))

def dual_Doppler(x1,x2,y1,y2,min_range,max_range):
    
    x=np.arange(-2000,2011,10)#[m] dual-Doppler x
    y=np.arange(-2000,2001,10)#[m] dual-Doppler y 
    
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
    
    #exclude ranges
    DD['range1']=((DD.x-x1)**2+(DD.y-y1)**2)**0.5
    DD['range2']=((DD.x-x2)**2+(DD.y-y2)**2)**0.5
      
    DD=DD.where((DD['range1']>min_range)*(DD['range1']<max_range)*\
                (DD['range2']>min_range)*(DD['range2']<max_range))
     
    return DD

def matrix_plt(x,y,f,cmap,vmin,vmax):
    '''
    Plot matrix with color and display values
    '''
    pc=plt.pcolor(x,y,f.T,cmap=cmap,vmin=vmin,vmax=vmax)
    for i in range(len(x)):
        for j in range(len(y)):
            if ~np.isnan(f[i,j]):
                ax.text(i,j, f"{f[i,j]:.1f}", 
                        ha='center', va='center', color='k', fontsize=10,fontweight='bold')
            
    return pc

#%% Inputs
source_nwtc=os.path.join(cd,'data/FC_assets.xlsx')#source nwtc sites
source_m2=os.path.join(cd,'data/nwtc.m2.b0/*nc')#source of M2 wind data
min_range=100#[m] minimum range
max_range=2000#[m] maximum range

bins_ws=np.array([0,2.5,5,7.5,10,15,25])#[m/s] bins in wind speed
bins_wd=np.arange(0,360,30)#[deg] bins in wind direction

#%% Initialization

#read data
FC=pd.read_excel(source_nwtc).set_index('Site')
M2=xr.open_mfdataset(glob.glob(source_m2))

#Cartesianize sites
FC['x_utm'],FC['y_utm'],FC['zone_utm1'],FC['zone_utm2']=utm.from_latlon(FC['Lat'].values, FC['Lon'].values)

#climatology
M2=M2.where(M2.WS_5m>0)
N=     stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='count',bins=[bins_ws,bins_wd])[0]
ws_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WS_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
wd_avg=stats.binned_statistic_2d(M2.WS_5m.values,M2.WD_5m.values,M2.WD_5m.values,statistic='mean', bins=[bins_ws,bins_wd])[0]
 
#dual-Doppler locations
x_target=FC[FC['DD target']=='Yes']['x_utm']
y_target=FC[FC['DD target']=='Yes']['y_utm']
x_source=FC[FC['DD source']=='Yes']['x_utm']
y_source=FC[FC['DD source']=='Yes']['y_utm']

#zeroing
err_u=np.zeros((len(x_source),len(x_source)))+np.nan
err_v=np.zeros((len(x_source),len(x_source)))+np.nan
err_ws=np.zeros((len(x_source),len(x_source)))+np.nan
err_wd=np.zeros((len(x_source),len(x_source)))+np.nan

#%% Main
for i_s1 in range(len(x_source)):
    for i_s2 in range(i_s1+1,len(x_source)):

        #dual-Doppler maps
        x_DD1=x_source[i_s1]
        x_DD2=x_source[i_s2]
        y_DD1=y_source[i_s1]
        y_DD2=y_source[i_s2]

        #dual-Doppler error
        DD=dual_Doppler(x_DD1, x_DD2, y_DD1, y_DD2,min_range,max_range)
       
        DD['N']=xr.DataArray(N/np.sum(N),    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})
        DD['ws_avg']=xr.DataArray(ws_avg,    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})
        DD['wd_avg']=xr.DataArray(wd_avg,    coords={'ws':(bins_ws[1:]+bins_ws[:-1])/2,'wd':(bins_wd[1:]+bins_wd[:-1])/2})
        
        DD['sigma_ws']=(((cosd(270-DD['wd_avg']))**2*DD['sigma_u']**2+\
                         (sind(270-DD['wd_avg']))**2*DD['sigma_v']**2)**0.5*DD['N']).sum(dim='ws').sum(dim='wd')
        DD['sigma_ws']=DD['sigma_ws'].where(~np.isnan(DD['sigma_u']))
            
        DD['sigma_wd']=(((sind(270-DD['wd_avg']))**2*DD['sigma_u']**2+\
                         (cosd(270-DD['wd_avg']))**2*DD['sigma_v']**2)**0.5/DD['ws_avg']*DD['N']).sum(dim='ws').sum(dim='wd')*180/np.pi
        DD['sigma_wd']=DD['sigma_wd'].where(~np.isnan(DD['sigma_u']))
            
        #evaluate at target points
        DD_interp=DD.interp(x=x_target,y=y_target)
        err_u[i_s1,i_s2]= np.mean(np.diag(DD_interp['sigma_u'].values))
        err_v[i_s1,i_s2]= np.mean(np.diag(DD_interp['sigma_v'].values))
        err_ws[i_s1,i_s2]=np.mean(np.diag(DD_interp['sigma_ws'].values))
        err_wd[i_s1,i_s2]=np.mean(np.diag(DD_interp['sigma_wd'].values))
            
        #plots
        #dual-Doppler error (u,v)
        plt.figure(figsize=(12,10))
        ax=plt.subplot(2,2,1)
        cf=plt.contourf(DD.x,DD.y,DD.sigma_u,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
        ax.set_aspect('equal')
        plt.ylabel('$y$ [m]')
        ax.set_xticklabels([])
        plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_u'].values),cmap='RdYlGn_r',vmin=0,vmax=10,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of $u$')
           
        ax=plt.subplot(2,2,2)
        cf=plt.contourf(DD.x,DD.y,DD.sigma_v,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_v'].values),cmap='RdYlGn_r',vmin=0,vmax=10,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of $v$')
        
        ax=plt.subplot(2,2,3)
        cf=plt.contourf(DD.x,DD.y,DD.sigma_ws,np.arange(0,10.1,0.1),cmap='RdYlGn_r',extend='both')
        ax.set_aspect('equal')
        plt.xlabel('$x$ [m]')
        plt.ylabel('$y$ [m]')
        plt.xticks(rotation=30) 
        plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_ws'].values),cmap='RdYlGn_r',vmin=0,vmax=10,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of wind speed')
        
        ax=plt.subplot(2,2,4)
        cf=plt.contourf(DD.x,DD.y,DD.sigma_wd,np.arange(0,91),cmap='RdYlGn_r',extend='both')
        ax.set_aspect('equal')
        plt.xlabel('$x$ [m]')
        plt.xticks(rotation=30) 
        ax.set_yticklabels([])
        plt.scatter(x_target,y_target,s=20,c=np.diag(DD_interp['sigma_wd'].values),cmap='RdYlGn_r',vmin=0,vmax=90,edgecolor='k')
        plt.plot([x_source[i_s1],x_source[i_s2]],[y_source[i_s1],y_source[i_s2]],'bs',markersize=5)
        plt.colorbar(cf,label=r'Error factor of wind direction [$^\circ$ s m$^{-1}$]')
        
        plt.savefig(os.path.join(cd,'figures',f'{x_source.index[i_s1]}-{x_source.index[i_s2]}.png'))
        plt.close()
        
#%% Output
err=err_u/np.nanmedian(err_u)+err_v/np.nanmedian(err_v)+err_ws/np.nanmedian(err_ws)+err_wd/np.nanmedian(err_wd)

sites=x_source.index  
SITES1,SITES2=np.meshgrid(sites,sites)
Output=pd.DataFrame({'Site1':SITES1.ravel(),'Site2':SITES2.ravel(),'Total error':err.ravel()})
Output.to_excel(os.path.join('data','dual-Doppler_opt.xlsx'))

#%% Plots
fig=plt.figure(figsize=(18,10))
ax=plt.subplot(2,2,1)
pc=matrix_plt(sites,sites, err_u, cmap='RdYlGn_r', vmin=0, vmax=10)
plt.title('Mean error factor of $u$')
plt.ylabel('Lidar 2 site')
ax.set_xticklabels([])
plt.title(r'Mean error factor of $u$')
ax=plt.subplot(2,2,2)
pc=matrix_plt(sites,sites, err_v, cmap='RdYlGn_r', vmin=0, vmax=10)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.title(r'Mean error factor of $v$')
ax=plt.subplot(2,2,3)
pc=matrix_plt(sites,sites, err_ws, cmap='RdYlGn_r', vmin=0, vmax=10)
plt.xlabel('Lidar 1 site')
plt.ylabel('Lidar 2 site')
plt.xticks(rotation=30) 
plt.title(r'Mean error factor of wind speed')
ax=plt.subplot(2,2,4)
pc=matrix_plt(sites,sites, err_wd, cmap='RdYlGn_r', vmin=0, vmax=90)
plt.xlabel('Lidar 1 site')
ax.set_yticklabels([])
plt.xticks(rotation=30) 
plt.title(r'Mean error factor of wind direction [$^\circ$ s m$^{-1}$]')
plt.tight_layout()
     