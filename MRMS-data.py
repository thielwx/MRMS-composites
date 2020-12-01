#!/usr/bin/env python
# coding: utf-8


# This program is created to pull data from raid and prepare it for the MRMS-processing.py script



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

# ### Function Land

#This function copies data from the MRMS raid
def MRMS_copier(case,start,end,var):
    
    #Making the save path if it doesn't exist yet
    savepath = '/localdata/cases/'+case+'/MRMS/'+var
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    #Command to get data on every minute ending in zero
    cmd = 'cp /raid/swat_archive/vmrms/CONUS/'+start+'/multi/'+var+'/00.50/'+start+'-???0* '+savepath
    p = sp.Popen(cmd,shell=True)
    p.wait()

    #Command to get data on every minute ending in 6
    cmd = 'cp /raid/swat_archive/vmrms/CONUS/'+start+'/multi/'+var+'/00.50/'+start+'-???6* '+savepath
    p = sp.Popen(cmd,shell=True)
    p.wait()

    #Command to get data from the final minute (held in the next day)
    cmd = 'cp /raid/swat_archive/vmrms/CONUS/'+end+'/multi/'+var+'/00.50/'+end+'-0000* '+savepath
    p = sp.Popen(cmd,shell=True)
    p.wait()
    
    #Command to unzip all of the gz files
    cmd = 'gunzip '+savepath+'/*.gz'
    p = sp.Popen(cmd,shell=True)
    p.wait()



def timestring(time):
    y = datetime.strftime(time,'%Y')
    j = datetime.strftime(time,'%j')
    h = datetime.strftime(time,'%H')
    M = datetime.strftime(time,'%M')
    d = datetime.strftime(time,'%d')
    m = datetime.strftime(time,'%m')
    newtime = y+m+d+'-'+h+M
    
    return newtime





# ### Work Zone



#===================================================
#  Run like this: python MRMS_data.py May/20190528 20190528 20190529 May
#===================================================

args = sys.argv
#args = ['May/20190528','20190528','20190529','May']

case = str(args[1])
start = str(args[2])
end = str(args[3])
name = str(args[4])

#Loop so its a per variable basis
variables = ['MESH','Reflectivity_-10C','MergedReflectivityQCComposite','ReflectivityAtLowestAltitude']
for i in variables:
    MRMS_copier(case,start,end,i)



