#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This script is designed to take the processed MRMS, GLM/ABI, and LSR data and combine it into one big dataframe


# In[16]:


import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os
import sys

# In[17]:
#===================================================
#  Run like this: python MRMS-post-process.py 2019/05/28 2019/05/29
#===================================================

#Setting up the dates and times to use for the file extensions
#args = ['potato','2019/05/21','2019/05/31']
args = sys.argv

start_date = args[1]
end_date = args[2]

datetime_list = pd.date_range(start=start_date,end=end_date,freq='D').to_pydatetime().tolist()

date_list = []
month_list = []
for i in datetime_list:
    date_list.append(i.strftime('%Y%m%d'))
    month_list.append(i.strftime('%B'))
    
time_index = np.arange(0,len(date_list),1)


# In[18]:


#Looping through on a per-case basis
for i in time_index:
    
    #Checking to see if the MRMS data exists, becuase if not I don't want to process it
    loc = '/localdata/cases/'+month_list[i]+'/'
    if os.path.exists(loc+date_list[i]+'/'+'MRMS'):
        print(datetime_list[i])
        
        #Loading in the data
        MRMS_data = pd.read_pickle(loc+date_list[i]+'/'+'MRMS/'+date_list[i]+'-MRMS.pkl')
        GOES_data = pd.read_pickle(loc+'data/'+date_list[i]+'.pkl')
        lsr_data  = pd.read_pickle(loc+'all_lsr_10min_STORM_DATA_timeinterp/lsr_'+date_list[i]+'_STORM_DATA_timeinterp.pkl')
        
        #Combining all of the data into a single dataframe and then reducing it
        all_data  = pd.concat((MRMS_data,GOES_data,lsr_data),axis=1)
        del(MRMS_data,GOES_data,lsr_data)
        all_data = all_data.loc[((all_data['MergedReflectivityQCComposite']>0)|(all_data['MESH'])|(all_data['Reflectivity_-10C'])|(all_data['ReflectivityAtLowestAltitude'])|(all_data['VIL']>0)|(all_data['VII']>0)),:]
        
        #Saving the file
        all_data.to_pickle(loc+'MRMS-combo-4/'+date_list[i]+'.pkl')
        print (loc+'MRMS-combo-4/'+date_list[i]+'.pkl')
        del(all_data)
        

#Combining all of the reduced data into one dataframe
# In[ ]:
april_loc = '/localdata/cases/April/'
may_loc = '/localdata/cases/May/'

may_files = os.listdir(may_loc+'MRMS-combo-4/')
april_files = os.listdir(april_loc+'MRMS-combo-4/')

may_df = pd.DataFrame()
for i in may_files:
    print (i)
    may_df = pd.concat((may_df,pd.read_pickle(may_loc+'MRMS-combo-4/'+i)),axis=0)
    
april_df = pd.DataFrame()
for i in april_files:
    print (i)
    april_df = pd.concat((april_df,pd.read_pickle(april_loc+'MRMS-combo-4/'+i)),axis=0)
    
april_df.to_pickle(april_loc+'MRMS-total-data-April-4.pkl')
may_df.to_pickle(may_loc+'MRMS-total-data-May-4.pkl')



