#!/usr/bin/env python
# coding: utf-8

# # Identification of focal conflicts

# This script identifies calls that are close in time and could thus be the same 
# call, recorded by multiple recording devices. These calls are pulled from the 
# dataset.
# 
# Necessary inputs are 
# - the labelfile of all calls (labelfile.csv)
# 
# Outputs are 
# 
# - the dictionary "candidates_matches.json", containing each call with potential matches as key and all its potential matches as values).
# 
# - Subset of labelfile, containing only the calls contained in the dictionary (either as key or value or both) is saved as "candidates_labelfile.csv" 
# 
# Both saved in current working directory. These outputs will be read by subsequent parts of the resolve_focal_conflicts pipeline.

# In[1]:


import pandas as pd
import os
import numpy as np
import json 
import datetime
import sys


# # Get server path

# In[1]:


#SERVER = input("Enter path to EAS server (e.g. /Volumes or //10.126.19.90 or /home/Documents/MPI_server: ")


# In[7]:


if not os.path.exists('server_path.txt'):
    SERVER = input("Enter path to EAS server (e.g. /Volumes or //10.126.19.90 or /home/Documents/MPI_server: ")
    with open('server_path.txt', 'w') as f:
        f.write("%s" % (SERVER))
        f.close()
else:
    f = open('server_path.txt', "r")
    SERVER = f.read().strip()
    f.close()


# In[3]:


if not os.path.exists(SERVER):
    print("Invalid server path: ", SERVER)
    exit()  
    
# If someone put a slash or backslash in last position
#if SERVER[-1:]=="/" or SERVER[-1:]=="\n":
#    SERVER = SERVER[:-1]


# In[30]:


# Save for later parts of the script
#with open('server_path.txt', 'w') as file:
#    file.write("%s" % (SERVER))
#file.close()


# In[4]:


# path to labelfile generated with 00_1_extract_calls
FULL_LABELFILE = SERVER + os.path.join(os.path.sep, 'EAS_shared',
                             'meerkat','working','processed',
                             'acoustic', 'extract_calls', 'labelfile.csv')


HOME = SERVER + os.path.join(os.path.sep, 'EAS_shared',
                             'meerkat','working','processed',
                             'acoustic', 'resolve_conflicts')
# Matching parameters

# search for calls with start times <=TIME_THRESH apart
TIME_THRESH= 0.1

# time factor threshold for how similar in duration two calls have to be to be flagged as possibly the same
DUR_FACTOR = 1.3 #(longer call can be max DUR_FACTOR as long as shorter call)


# In[5]:


def get_datetime(timestring):
    """
    Function that gets datetime object from date and time string
    string must match one of the given patterns

    Parameters
    ----------
    timestring : String
                 some string containing a date and time

    Returns
    -------
    result : datetime object
             the time as datetime object

    Example
    -------
    >>> dt = get_datetime("2017-08-06 07:49:24")
    
    """ 
    time_patterns = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']
    for pattern in time_patterns:
        try:
            return datetime.datetime.strptime(timestring, pattern)
        except:
            pass

    print("Date is not in expected format:", timestring)
    
    sys.exit(0)
    

def get_timediff(time1, time2):
    """
    Function that gets time difference between two datetime objects
    in seconds 

    Parameters
    ----------
    time1 : A datetime object
    time2 : A datetime object

    Returns
    -------
    Numeric
    Time difference in seconds     
    """ 
    duration = time1-time2
    return abs(duration.total_seconds())


# # Read in files

# In[6]:


print("Reading in labelfile...")


# In[7]:


labelfile = pd.read_csv(FULL_LABELFILE, sep="\t")
print(labelfile.shape)


# In[5]:


#labelfile.columns


# In[8]:


# remove zero duration calls
labelfile = labelfile.loc[labelfile.duration_s>0,:]
print(labelfile.shape)
# remove non-calls
labelfile = labelfile.loc[labelfile.isCall==1,:]
print(labelfile.shape)
# remove nonfocal labelled calls
labelfile = labelfile.loc[labelfile.focalType!="NF",:]
print(labelfile.shape)


# # Identify potential matches

# In[9]:


print("Identifying potential matches ...")


# In[10]:


# Reduce amount of computation by checking focal-nonfocal pairs per day.
all_dates = sorted(list(set(labelfile.date)))
match_dict = {}
files_dict = {}

for date in all_dates:
    print("Processing date: ", date, end='\r')
    matches = {}
    sub_df = labelfile.loc[labelfile.date==date,:]
      
    # Calculare time distance between all
    
    callIDs = list(sub_df.callID_new)
    durations = list(sub_df.duration_s)
    indvs = list(sub_df.ind)
    
    date_time = [get_datetime(x) for x in sub_df['t0GPS_UTC']]
    time_diff_mat = np.zeros((len(date_time), len(date_time)))
    
    for i in range(len(date_time)):
        for j in np.arange(i,len(date_time)):
            time_diff_mat[i,j] = get_timediff(date_time[i],date_time[j])
            time_diff_mat[j,i] = time_diff_mat[i,j]
      
    
    # Don't want to find pair of call with itself, so I set the diagonal above threshold
    for i in range(len(date_time)):
        time_diff_mat[i,i] = 999
    
    # Select all pairs below threshold:
    potential_matches = np.nonzero(time_diff_mat<=TIME_THRESH)
    
    # save potential pairs in two vectors x,y
    x = potential_matches[0]
    y = potential_matches[1]
    
    # dict by callID
    for i, j in zip(x,y):
        
        # select only if of different individuals
        if (indvs[i]!=indvs[j]):
            
            # select only if duration diff is below threshold
            durs = [durations[i], durations[j]]            
            if durs[np.argmax(durs)] <= durs[np.argmin(durs)]*DUR_FACTOR:
                
                # Note: this allows for "double entry" of a call pair, i.e. dict[x] = y AND dict[y] = x
                match = [callIDs[i], callIDs[j]]
                
                # add pair to dictionary if not yet in dictionary
                if match[0] not in matches.keys():
                    matches[match[0]] = [match[1]]                   
                else:
                    matches[match[0]] = matches[match[0]]+[match[1]]
                    
    
    files_we_need = []
    
    for key in matches.keys():
        files_we_need.append(key)
        for val in matches[key]:
            files_we_need.append(val)
    files_we_need = list(set(files_we_need))
    
    match_dict[date] = matches
    files_dict[date] = files_we_need


all_files_we_need = []
all_matches = {}

for key in files_dict.keys():
    all_files_we_need = all_files_we_need + files_dict[key]
    for m_key, m_val in zip(match_dict[key].keys(), match_dict[key].values()):
        all_matches[m_key] = m_val
        
print(len(all_matches)) 
print(len(all_files_we_need))


# # Save results

# In[11]:


candidates_labelfile = labelfile.loc[labelfile['callID_new'].isin(all_files_we_need)]
candidates_labelfile.reset_index(inplace=True, drop=True)

candidates_labelfile_out = os.path.join(os.path.sep, HOME,'candidates_labelfile.csv')
candidates_labelfile[['callID_new', 
                      'ind', 
                      'focalType', 
                      'entryName', 
                      'wavFileName', 
                      'start_s', 
                      'duration_s',
                      'x_emitted',
                      'y_emitted']].to_csv(candidates_labelfile_out, sep="\t", index=False)

candidates_matches_out = os.path.join(os.path.sep, HOME,'candidates_matches.json')
with open(candidates_matches_out, "w") as outfile:  
    json.dump(all_matches, outfile)  


# In[12]:


print("Done")


# In[ ]:




