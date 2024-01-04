#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'This script takes data from */localdata/TonyData/* and put it into their respective case days'


# In[13]:


import os
import sys
import subprocess as sp
import pandas as pd
from datetime import datetime
import datetime
import numpy as np


# In[ ]:


#=========================================================
#Run like this: python MRMS-data-mover 2019/04/20 2019/04/29
#=========================================================


# In[7]:


args = sys.argv
#args = ['potato','2019/04/20','2019/04/29']

start_date = args[1]
end_date = args[2]


# In[8]:


datetime_list = pd.date_range(start=start_date,end=end_date,freq='D').to_pydatetime().tolist()


# In[34]:


start_list = []
end_list = []
smonth_list = []
emonth_list = []

timedelta = datetime.timedelta(days=1)

for i in datetime_list:
    
    timestr = i.strftime('%Y%m%d')
    start_list.append(timestr)
    
    endtime = i + timedelta
    endstr = endtime.strftime('%Y%m%d')
    end_list.append(endstr)
    
    monthstr = i.strftime('%B')
    smonth_list.append(monthstr)
    
    monthstr = endtime.strftime('%B')
    emonth_list.append(monthstr)
    
indexes = np.arange(0,len(start_list))


# In[19]:


t_string = '/localdata/TonyData/'


# In[31]:


def to_cases(start,smonth,var): 
    dfrom = t_string + var + '/' + start+'-*.netcdf '
    dto = '/localdata/cases/'+smonth+'/'+start+'/MRMS/'+var+'/'
    cmd = 'mv '+dfrom+dto
    
    print (cmd)
    p = sp.Popen(cmd,shell=True)
    p.wait()
    
    return 0


# In[42]:


def kickback(start,end,smonth,emonth,var):
    dfrom = '/localdata/cases/'+emonth+'/'+end+'/MRMS/'+var+'/'+end+'-0000*.netcdf '
    dto = '/localdata/cases/'+smonth+'/'+start+'/MRMS/'+var+'/'
    cmd = 'mv '+dfrom+dto
    print (cmd)
    
    p = sp.Popen(cmd,shell=True)
    p.wait()
    
    return 0


# In[41]:


print ('MOVING DATA FROM /LOCALDATA/TONYDATA/')
for i in indexes:
    for j in ['VIL','VII']:
        to_cases(start_list[i],smonth_list[i],j)

print ('MOVING THE LAST FILE TO THE PREVIOUS DAY')
for i in indexes:
    for j in ['VIL','VII']:
        kickback(start_list[i],end_list[i],smonth_list[i],emonth_list[i],j)


# In[ ]:




