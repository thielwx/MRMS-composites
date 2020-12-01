#!/usr/bin/env python
# coding: utf-8
#This script takes in the downloaded MRMS data, transforms it into geostatioanry coordiates
#from GOES-16, and outputs the data as a dataframe in the same format as the ABI-GLM datasets

# In[1]:


import numpy as np
import pandas as pd
from scipy import spatial
from pyproj import Proj
import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import sys
import os
import subprocess as sp
import glob
import pyresample as pr


# # Funciton Land

# In[2]:


#This funciton does the dirty work of loading in the x,y data and creating the file list
#Keeing the work zone clean
def setup(case,start,end):
    #loading in the 20 km grids used for the ABI and GLM comparisons
    x20 = np.load('/localdata/coordinates/20km_x.npy')
    y20 = np.load('/localdata/coordinates/20km_y.npy')
    xx20,yy20 = np.meshgrid(x20,y20)
    xf = xx20.flatten()
    yf = yy20.flatten()
    
    #Getting the time list together for every minute ending in 0 and 4
    stime = datetime.strptime(start,'%Y%m%d')
    etime = datetime.strptime(end,'%Y%m%d')
    tlist = pd.date_range(start=stime,end=etime,freq='T').to_pydatetime().tolist()
    tlist6 = tlist[6::10]
    tlist5 = tlist[5::10]
    tlist0 = tlist[0::10]
    tlist = sorted(np.concatenate((tlist0,tlist6))) #We're pulling data from every minute ending in 0 and 6
    tlist_df = sorted(np.concatenate((tlist0,tlist5))) #For the dataframe, using 0 and 5 minutes as the output
    index = np.arange(0,len(tlist))
    
    file_list = []
    
    for i in index:
        h = datetime.strftime(tlist[i],'%H') #hour
        M = datetime.strftime(tlist[i],'%M') #minute
        y = datetime.strftime(tlist[i],'%Y') #year
        m = datetime.strftime(tlist[i],'%m') #month
        d = datetime.strftime(tlist[i],'%d') #day

        file_str = y+m+d+'-'+h+M+'*'+'.netcdf'
        file_list = np.append(file_list,file_str)
            
    return x20,y20,xf,yf,tlist,tlist_df,file_list,index
        


# In[3]:


#Using the glob function to see if the people are there
def file_check(file,case,start,var):
    loc = '/localdata/cases/'+case+'/MRMS/'+var+'/'
    #print (loc+file)
    fcheck = glob.glob(str(loc+file))
    #print (fcheck)
    return fcheck,loc


# In[4]:


#This function takes in the data from glm16_proj, clusters it, and samples the max value
#This uses a brute force method (regarded a truth)
def MRMS_cluster1(x,y,var,x20,y20):
    #Adding extra values to the end of the array so that we can still cluster at the edge of the grid
    x20 = np.concatenate(([-100],x20,[100]))
    y20 = np.concatenate(([100],y20,[-100]))
    
    #Creating a grid of average values so we don't have to call np.avg for each point
    #For xavg, matching indicies are to the left of the desired point, +1 index is to the right
    #For yavg, matching indicies are above the desired point, +1 index is below
    xavg = np.average((x20[1:],x20[:-1]),axis=0)
    yavg = np.average((y20[1:],y20[:-1]),axis=0)
    

    #Setting our index to iterate through
    x20_index = np.arange(0,len(x20),1)
    y20_index = np.arange(0,len(y20),1)
    
    grid = np.ones((len(x20)-2,len(y20)-2)) * np.nan #Grid but with the extra points taken out
    
    #For loops that only refer to the indicies of real data and not at the end
    for i in x20_index[1:-1]:
        for j in y20_index[1:-1]:
            #Finding the points that lie between the x and y bounds
            locs = np.where((x<xavg[i])&(x>=xavg[i-1])&(y<yavg[j-1])&(y>=yavg[j]))[0]
            
            #Changing the values of the array to the max of the clustered points
            if len(locs)>0:
                var_max = np.max(var[locs])
                grid[i-1,j-1] = var_max
    
    return grid.T.flatten(), xavg, yavg  #Transposing to fit with grid
            


# In[5]:


#This function clusters the data to the 20 km GOES-16 grid using a KD-tree to sample the max value
#The radius of influence is set to 15 km == 15000 m
#MUCH faster than MRMS_cluster

def MRMS_cluster2(x20,y20,var,lat,lon):
    
    MRMS_swath = pr.SwathDefinition(lons=lon,lats=lat)
    lon20,lat20 = latlon(x20,y20)
    grid20_swath = pr.SwathDefinition(lons=lon20,lats=lat20)

    valid_input_index, valid_output_index, index_array, distance_array = pr.kd_tree.get_neighbour_info(
        source_geo_def=MRMS_swath,target_geo_def=grid20_swath,radius_of_influence=15000,neighbours=800)
    #The valid_input_index gives which MRMS values fall in the domain (I think?) - T/F array
    #The valid_output_index gives which 20 km grid points have values to interpret into the grid
    #The index_array says where they are in the 1-D MRMS array
    #The distance_array says how close they are (likely used for nearest neighbor)



    #Making the arrays we'll use to trick the kd_tree.get_sample_from_neighbor_info
    #Valid_input_index and valid_output_index won't change
    #The index array is now one dimensional and filled with the lenght of the MRMS data (means no data found)
    imposter_index_array = np.ones(index_array.shape[0],dtype=int)*int(len(valid_input_index))
    #We don't care about the distance, so just filling with ones
    imposter_distance_array = np.ones(distance_array.shape[0])

    #Giving the indicies in the grid where data exists (min value is not where there is not MRMS data)
    data_locs = np.where( np.min(index_array,axis=1) != len(valid_input_index) )[0]

    #For loop that goes through where the imposter index array needs changed
    for i in data_locs:
        #So at each point where data exists...
        #   -find where the data exists in the individual index array for each 20km grid point
        point_locs = np.where(index_array[i,:] != len(valid_input_index))[0]
        point_indicies = index_array[i,point_locs]

        #Getting the max value and finding it's index
        val_max = np.max(var[point_indicies])
        a = point_indicies[np.where(var[point_indicies]==val_max)]

        #Placing the index in the imposter_index_array
        imposter_index_array[i] = int(a[0])

    #Using resampling using kd-tree with imposter_index_array to force the nearest neighbor to be the max value
    output =  pr.kd_tree.get_sample_from_neighbour_info(resample_type='nn',
                                              output_shape=grid20_swath.shape,
                                              data=var,
                                              valid_input_index = valid_input_index,
                                              valid_output_index = valid_output_index,
                                              index_array = imposter_index_array)

    truther = output == 0
    output[truther] = np.nan
    
    return output.flatten()


# In[6]:


#More or less copied from fn_master.py

def latlon(X,Y):    
    sat_h = 35786023.0
    sat_lon = -70
    sat_sweep = 'x'
    
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    YY, XX = np.meshgrid(Y*sat_h, X*sat_h)
    lons, lats = p(XX, YY, inverse=True)
    
    return lons.T, lats.T


# In[7]:


#This function takes in the lat and lon coordiates from MRMS data and turns them into geostationary projections
#Basically the inverse of the latlon function in fn_master.py
def g16_proj(lon,lat):
    sat_h = 35786023.0
    sat_lon = -70
    sat_sweep = 'x'
    
    p = Proj(proj='geos',h=sat_h,lon_0=sat_lon,sweep=sat_sweep)
    x,y = p(lon,lat,inverse=False)
    
    return x/sat_h,y/sat_h


# In[8]:


#This function loads in the MRMS data and put it into the 20 km grid used for the ABI/GLM data

def mrms_processor(file,var,loc,x20,y20):
    #loading in the data from the MRMS netcdf file
    dset = nc.Dataset(file,'r')
    x_pix = dset.variables['pixel_x'][:] #Pixel locations (indicies) for LATITUDE
    y_pix = dset.variables['pixel_y'][:] #Pixel locations (indicies) for LONGITUDE
    data = dset.variables[var][:]
    
    u_lat = dset.Latitude #Upper-most latitude
    l_lon = dset.Longitude #Left-most longitude
    
    #Creating the arrays for the lat and lon coordinates
    y = dset.dimensions['Lat'].size #3500
    x = dset.dimensions['Lon'].size #7000
    lat = np.arange(u_lat, u_lat-(y*0.01),-0.01) #Going from upper to lower
    lon = np.arange(l_lon, l_lon+(x*0.01),0.01) #Going from left to right
    
    #Using the pixel indicides to get the pixel latitudes and longitudes
    lat = lat[x_pix] #Remember x_pixel represents LATITUDE
    lon = lon[y_pix] #Remember y_pixel represent LONGITUDE

    #Removing all data west of 103W and also any false data
    locs = np.where((lon>=-103)&(data>0))[0]
    lon = lon[locs]
    lat = lat[locs]
    data = data[locs]
    
    #Turning the lat,lon coordinates to geostationary projection coordinates from goes-16
    #x_sat,y_sat = g16_proj(lon,lat)
    
    #Clustering the data into 20 km bins that match the ABI/GLM dataset
    if len(data)>0: #If statement accounts for corrupted MRMS files that only consist of the fill value (therefore no data)
        #grid = MRMS_cluster1(x_sat,y_sat,data,x20,y20) #the old one, often referred to as the 'brute force method'
        grid = MRMS_cluster2(x20,y20,data,lat,lon) #kd-tree method that runs over 100 times faster
        print (var+' data accepted')
    else:
        grid = np.ones(len(xf))*np.nan
        print ('*****BAD DATA '+var+'*****')
    return grid


def mrms_processor2(file,var,loc,x20,y20):
    dset = nc.Dataset(file,'r')
    data = dset.variables[var][:,:]
    u_lat = dset.Latitude #Upper-most latitude
    l_lon = dset.Longitude #Left-most longitude

    #Creating the arrays for the lat and lon coordinates
    y = dset.dimensions['Lat'].size #3500
    x = dset.dimensions['Lon'].size #7000
    lat = np.arange(u_lat, u_lat-(y*0.01),-0.01) #Going from upper to lower
    lon = np.arange(l_lon, l_lon+(x*0.01),0.01) #Going from left to right

    glon,glat = np.meshgrid(lon,lat) #Creating the grid of lat and lon data points

    data = data.flatten()
    glon = glon.flatten()
    glat = glat.flatten()

    where = (data>0)&(glon>=-103)

    lon = glon[where]
    lat = glat[where]
    data = data[where]

    #Turning the lat,lon coordinates to geostationary projection coordinates from goes-16
    #x_sat,y_sat = g16_proj(lon,lat)
    
    #Clustering the data into 20 km bins that match the ABI/GLM dataset
    if len(data)>0: #If statement accounts for corrupted MRMS files that only consist of the fill value (therefore no data)
        #grid = MRMS_cluster1(x_sat,y_sat,data,x20,y20) #the old one, often referred to as the 'brute force method'
        grid = MRMS_cluster2(x20,y20,data,lat,lon) #kd-tree method that runs over 100 times faster
        print (var+' data accepted')
    else:
        grid = np.ones(len(xf))*np.nan
        print ('*****BAD DATA '+var+'*****')
    return grid

# In[9]:


def driver(file,loc,var,x,y,xf,yf):
    #If there is a data file, run the procedure to find it
    if (len(file)>0)&((var=='VII')|(var=='VIL')):
        data = mrms_processor2(file[0],var,loc,x,y)
    elif len(file)>0:
        data = mrms_processor(file[0],var,loc,x,y)
    #If there isn't, fill it full of nans
    else:
        data = np.ones(len(xf))*np.nan
        print ('*****NO DATA '+var+'*****')
        
    return data


# In[14]:


def df_maker(mesh,iso_dbz,comp_dbz,rala,vii,vil,t,t_df,xf,yf):
    
    y = datetime.strftime(t_df,'%Y')
    h = datetime.strftime(t_df,'%H')
    d = datetime.strftime(t_df,'%d')
    M = datetime.strftime(t_df,'%M')
    m = datetime.strftime(t_df,'%m')
    
    t_str = y+m+d+'-'+h+M #Creating the time index for the outer index
    
    time = np.broadcast_to(t_str,len(xf))
    
    dummy = np.ones(len(xf))*np.nan #A dummy array we use to prefill the dataframes
    df = pd.Series(dummy,name='Dummy')
    mi = pd.MultiIndex.from_arrays([time,xf,yf])
    df = df.reindex(mi)
    
    #Dumping all of the data into the array
    newdf = pd.DataFrame({'Dummy':df,
                         'MESH':mesh,
                         'Reflectivity_-10C':iso_dbz,
                         'MergedReflectivityQCComposite':comp_dbz,
                         'ReflectivityAtLowestAltitude':rala,
                         'VII':vii,
                         'VIL':vil})
    
    return newdf.drop(columns='Dummy')


# # Work Zone

# In[12]:


#===================================================
#  Run like this: python MRMS-processing.py May 20190528 20190529 May
#===================================================

args = sys.argv
#args = ['potato','test-data/20190520','20190520','20190521','test-data']

case = str(args[1])
start = str(args[2])
end = str(args[3])
name = str(args[4])
variables = ['MESH','Reflectivity_-10C','MergedReflectivityQCComposite','ReflectivityAtLowestAltitude','VII','VIL']

save_loc = '/localdata/cases/'+case+'/MRMS/'

#Setting up the variables we'll be using later (keeps the code down here a little cleaner)
x, y, xf, yf, t, t_df, files, index = setup(case,start,end)

#Setting up the multiindexed dataframe to append to
A = ['Null']
df = pd.Series([np.nan])
mi = pd.MultiIndex.from_arrays([A,A,A])
df = df.reindex(mi)

#Looping on the per timestamp basis
for i in index[1:]: #Need to skip the first one at 0000Z on the starting day
    
    #Checking for the files in case there's data gaps (stuff happens)
    mesh_file, mesh_loc = file_check(files[i],case,start,variables[0])
    iso_dbz_file, iso_dbz_loc = file_check(files[i],case,start,variables[1])
    comp_dbz_file, comp_dbz_loc = file_check(files[i],case,start,variables[2])
    rala_file, rala_loc = file_check(files[i],case,start,variables[3])
    vii_file, vii_loc = file_check(files[i],case,start,variables[4])
    vil_file, vil_loc = file_check(files[i],case,start,variables[5])
    
    print (t_df[i])

    #Getting the data from the files and transforming it into the geostationary grids
    mesh_data = driver(mesh_file,mesh_loc,variables[0],x,y,xf,yf)
    iso_dbz_data = driver(iso_dbz_file,iso_dbz_loc,variables[1],x,y,xf,yf)
    comp_dbz_data = driver(comp_dbz_file,comp_dbz_loc,variables[2],x,y,xf,yf)
    rala_data = driver(rala_file,rala_loc,variables[3],x,y,xf,yf)
    vii_data = driver(vii_file,vii_loc,variables[4],x,y,xf,yf)
    vil_data = driver(vil_file,vil_loc,variables[5],x,y,xf,yf)
    
    #Putting the data all into a dataframe that matches the GLM one
    df_combo = df_maker(mesh_data,iso_dbz_data,comp_dbz_data,rala_data,vii_data,vil_data,t[i],t_df[i],xf,yf)
    
    df = pd.concat((df,df_combo),axis=0,sort=True)
    
df = df.drop(index=['Null'],columns=0)

#Saving the dataframe
if not os.path.exists(save_loc):
    os.makedirs(save_loc)  
df.to_pickle(save_loc+start+'-MRMS.pkl')

