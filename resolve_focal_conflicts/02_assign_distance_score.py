#!/usr/bin/env python
# coding: utf-8

# # Assign distance score
# 
# Script to assign distance scores to call pairs using spectrogram comparison. 
# Requirements:
# - "candidates_matches.json", containing all potential matches (generated with 01_identify_focal_conflicts)
# - "candidates_labelfile.csv" of all calls involved in a match (generated with 01_identify_focal_conflicts)
# - "txts/" folder containing audio data for all calls (generated with 00_1_extract_calls)
# 
# Output:
# - a csv file containing all pairs of calls and their respective spectrogram distance, intensity and physical distance scores ("f_nf.csv"), saved in EAS_shared/.../resolve_conflicts.
# - may update "candidates_labelfile.csv" and "candidates_matches.json" if audio for calls can't be found

# ## Part 1: Get audio and spectrograms

# In[1]:


import pandas as pd
import os
import librosa
import sys
import numpy as np
import json
from scipy import stats
import math
from scipy.signal import butter, lfilter


# In[2]:


f = open('server_path.txt', "r")
SERVER = f.read().strip()
f.close()

HOME = SERVER + os.path.join(os.path.sep, 'EAS_shared',
                             'meerkat','working','processed',
                             'acoustic', 'resolve_conflicts')

# location of candidate files generated with 01_identify_focal_conflicts
CANDIDATES_MATCHES = os.path.join(os.path.sep, HOME,'candidates_matches.json')
CANDIDATES_LABELFILE = os.path.join(os.path.sep, HOME,'candidates_labelfile.csv')

# location of audio txt files generated with 00_1_generate_call_txts
TXT_PATH = SERVER + os.path.join(os.path.sep, 'EAS_shared',
                             'meerkat','working','processed',
                             'acoustic', 'extract_calls', 'txts')


# In[3]:


# Spectrogramming parameters
FFT_WIN = 0.03 # FFT_WIN*samplerate = length of fft/n_fft (number of audio frames that go in one fft)
FFT_HOP = FFT_WIN/8 # FFT_HOP*samplerate = n of audio frames between successive ffts
N_MELS = 40 # number of mel bins
WINDOW = 'hann' # each frame of audio is windowed by a window function (its length can also be
# determined and is then padded with zeros to match n_fft. we use window_length = length of fft
FMAX = 4000


# In[4]:


def read_wav_txt(filename):    
    """
    Function that reads audio data and sr from audio
    saved in txt format

    Parameters
    ----------
    data: String
          path to txt file
          
    Returns
    -------
    data : 1D np.array
           Raw audio data (Amplitude)
           
    sr: numeric (Integer)
        Samplerate (in Hz)
    """
    data = np.asarray([0])
    sr = 0
    
    try:
        f = open(filename, 'r')
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        sr = int(lines[0].split(':')[1])
        data = np.asarray([float(x) for x in lines[1:]])
        
        f.close()
        
    except Exception:
        print("No such file or directory: ", filename)
        pass
    return data, sr


def generate_mel_spectrogram(data, rate, n_mels, window, fft_win , fft_hop):
    
    """
    Function that generates mel spectrogram from audio data using librosa functions

    Parameters
    ----------
    data: 1D numpy array (float)
          Audio data
    rate: numeric(integer)
          samplerate in Hz
    n_mels: numeric (integer)
            number of mel bands
    window: string
            spectrogram window generation type ('hann'...)
    fft_win: numeric (float)
             window length in s
    fft_hop: numeric (float)
             hop between window start in s 

    Returns
    -------
    result : 2D np.array
             Mel-transformed spectrogram

    Example
    -------
    >>> 
    
    """
    n_fft  = int(fft_win * rate) 
    hop_length = int(fft_hop * rate) 
        
    s = librosa.feature.melspectrogram(y = data ,
                                       sr = rate, 
                                       n_mels = n_mels , 
                                       fmax = FMAX, 
                                       n_fft = n_fft,
                                       hop_length = hop_length, 
                                       window = window, 
                                       win_length = n_fft)

    spectro = librosa.power_to_db(s, ref=np.max)

    return spectro


# In[26]:


labelfile = pd.read_csv(CANDIDATES_LABELFILE, sep="\t")
labelfile.shape


# In[7]:


print("Fetching audio data for ", labelfile.shape[0], " calls...")


# In[27]:


audios_we_need = [os.path.join(os.path.sep, TXT_PATH, x+'.txt') for x in labelfile.callID_new]
raw_audio,samplerate_hz = map(list,zip(*[read_wav_txt(x) for x in audios_we_need]))

labelfile['raw_audio'] = raw_audio
labelfile['samplerate_hz'] = samplerate_hz


# In[9]:


# If there are any calls where no audio data was found, remove these from the labelfile and from matches

# Which IDs are missing?
missing = []
for audio, callID in zip(labelfile.raw_audio, labelfile.callID_new):
    if audio.shape[0]==1:
        missing.append(callID)

if len(missing)!=0:
    # remove missing from labelfile
    labelfile = labelfile.loc[~(labelfile['callID_new'].isin(missing)),:]
    
    # remove missing from matches
    with open(CANDIDATES_MATCHES, "r") as file:
        cand_matches = json.load(file)
    
    matches = {}    

    # remove the missing calls in the keys
    for key in cand_matches.keys():
        if key not in missing:
            # then remove any missing calls that are present as match partners
            match_partners = [p for p in cand_matches[key] if p not in missing]
            if len(match_partners)!=0:
                matches[key] = match_partners

    # save corrected candidate_matches and labelfile
    with open(CANDIDATES_MATCHES, "w") as outfile:  
        json.dump(matches, outfile) 
    
    labelfile.to_csv(CANDIDATES_LABELFILE, sep="\t", index=False)  


# In[10]:


print("Found audio data for ", labelfile.shape[0], " calls")


# In[28]:


# Generate spectrograms

spectrograms = labelfile.apply(lambda row: generate_mel_spectrogram(row['raw_audio'],
                                                                    row['samplerate_hz'],
                                                                    N_MELS,
                                                                    WINDOW,
                                                                    FFT_WIN,
                                                                    FFT_HOP), 
                               axis=1)


labelfile['spectrograms'] = spectrograms

denoised = [(spectrogram - np.median(spectrogram, axis=0)) for spectrogram in labelfile['spectrograms']]
labelfile['denoised_spectrograms'] = denoised


# ## Part 2: Assign distance score

# In[12]:


from pandas.core.common import flatten
from scipy import stats
import random


# In[13]:


# Template matching parameters

N_RANDOM_SHUFFLE = 10 # times to randomly shuffle spectrogram for normalization
MIN_OVERLAP = 0.9 # short spectrogram has to have at least MIN_OVERLAP with longer spectrogram
MAX_F_SHIFT = 0 # max frequency shift allowed when comparing spectrograms (in mel bins)
                # Left this in here for the future, but since it's set to zero, I am not allowing 
                # frequency shift at the moment
N_MELS=40 # N_Mels present
MEL_BINS_REMOVED_LOWER = 5 # remove lowest MEL_BINS_REMOVED_LOWER mel bins from 
                           # spectrograms (~all below 300 Hz), probably noise
MEL_BINS_REMOVED_UPPER = 5 # remove upmost MEL_BINS_REMOVED_UPPER mel bins from
                           # spectrograms (~all above 3 kHz), probably noise


# ## Read in data

# In[14]:


# dictionary of calls and their potential conflicting partners

with open(CANDIDATES_MATCHES, "r") as file:  
    matches = json.load(file)
len(matches)


# In[15]:


# list containing all pairs of calls
# (non-redundant, i.e. does not contain both [x,y] and [y,x], but only [x,y])

all_pairs = []

for key in matches.keys():
    for val in matches[key]:
        if [val,key] not in all_pairs:
            all_pairs.append([key, val])
        
print(len(all_pairs))


# # Calculate spectrogram dists

# In[16]:


def spec_dist(s_1, s_2):
    """
    Basic spectrogram distance function

    Parameters
    ----------
    s_1, s_2: 2D np.arrays
              the two spectrograms to compare
          
    Returns
    -------   
    norm_dist: numeric (Float)
               Squared error between specs s_1, s_2
               normalized to randomized sq. error 
               between s_1, s_2
    """
    dist = np.sum((np.subtract(s_1, s_2)*np.subtract(s_1, s_2)), axis=None)
    
    # Normalize to random shuffling
    random_dist = calc_random_dist(s_1,s_2)
    norm_dist = dist / random_dist 
    
    return norm_dist

def calc_random_dist(s_1, s_2):
    """
    Helper for basic spectrogram distance function
    Calculates randomized sq error between two
    spectrograms s_1, s_2 by shuffling s_1

    Parameters
    ----------
    s_1, s_2: 2D np.arrays
              the two spectrograms to compare         
    Returns
    -------   
        numeric (Float)
        Randomized sq. error between s_1, s_2
    """
    dists = []
    s_1_shuffled = np.copy(s_1)
    for i in range(N_RANDOM_SHUFFLE):
        np.random.shuffle(s_1_shuffled)
        dists.append(np.sum((np.subtract(s_1_shuffled, s_2)*np.subtract(s_1_shuffled, s_2)), axis=None)) 
    return(np.mean(dists))   

def calc_mindist(spec_a, spec_b):
    """
    Calculate min distance between two specs s_1, s_2
    by shifting specs against each other along time and 
    freq axis with certain global constraints

    Parameters
    ----------
    spec_a, spec_b: 2D np.arrays
                    the two spectrograms to compare         
    Returns
    -------   
    min_dist: numeric (Float)
              Minimum distance (produced by the best 
              overlap)
    """
    # Find the bigger spec
    spec_list = [spec_a, spec_b]
    spec_lens = [s.shape[1] for s in spec_list]
    
    if spec_a.shape[1]==spec_b.shape[1]:
        spec_s = spec_a
        len_s = spec_s.shape[1]
        spec_l = spec_b
        len_l = len_s
    else:
        spec_s = spec_list[np.argmin(spec_lens)] # shorter spec
        len_s = np.min(spec_lens)
        spec_l = spec_list[np.argmax(spec_lens)] # longer spec
        len_l = np.max(spec_lens)

    # define start position for time shifting
    # based on MIN_OVERLAP
    min_overlap_frames = int(MIN_OVERLAP * len_s)
    start_timeline = min_overlap_frames-len_s
    max_timeline = len_l - min_overlap_frames

    distances = []

    # shift short spec across longer, one time frame
    # at a time. Only compare the overlap section
    for timeline_p in range(start_timeline, max_timeline+1):
        # Select full specs or only a subset (start_col to end_col), 
        # depending on any of the three cases:
        
        # 1) mismatch on left side
        if timeline_p < 0:
            start_col_l = 0
            len_overlap = len_s - abs(timeline_p)
            end_col_l = start_col_l + len_overlap

            end_col_s = len_s # until the end
            start_col_s = end_col_s - len_overlap

        # 2) mismatch on right side
        elif timeline_p > (len_l-len_s):
            start_col_l = timeline_p
            len_overlap = len_l - timeline_p
            end_col_l = len_l

            start_col_s = 0
            end_col_s = start_col_s + len_overlap

        # 3) no mismatch on either side
        else:
            start_col_l = timeline_p
            len_overlap = len_s
            end_col_l = start_col_l + len_overlap

            start_col_s = 0
            end_col_s = len_s 
        
        s_s = spec_s[:,start_col_s:end_col_s] 
        s_l = spec_l[:,start_col_l:end_col_l] 
        distances.append(spec_dist(s_s, s_l))
        
        # Now do frequency shift
        
        # move smaller spec UP
        for fshift in range(1,MAX_F_SHIFT+1):
            s_s_fshifted = s_s[fshift:,:] # remove lowest freq row(s) from s_s
            s_l_fshifted = s_l[:-fshift,:] # remove highest freq row(s) from s_l
            
            distances.append(spec_dist(s_s_fshifted, s_l_fshifted))
        
        # move smaller spec DOWN
        for fshift in range(1,MAX_F_SHIFT+1):
            s_s_fshifted = s_s[:-fshift,:] # remove highest freq row(s) from s_s
            s_l_fshifted = s_l[fshift:,:]  # remove lowest freq row(s) from s_l
            
            distances.append(spec_dist(s_s_fshifted, s_l_fshifted))            
        

    min_dist = np.min(distances)
    return min_dist

def preprocess_spec(spec):
    """
    Do some transformations with denoised mel-spectrogram 
    to improve information content for template matching

    Parameters
    ----------
    spec: 2D np.array
          a denoised mel-spectrogram      
    
    Returns
    -------   
    spec: 2D np.array
          the transformed, denoised mel-spectrogram         
    """
    
    # Remove MEL_BINS_REMOVED_UPPER and -LOWER mels
    spec = spec[MEL_BINS_REMOVED_LOWER:(N_MELS-MEL_BINS_REMOVED_UPPER),:]

    # Z-score normalize
    spec = stats.zscore(spec, axis=None)
    
    # Cap vals > 3 STD higher than mean (intense=intense)
    spec = np.where(spec > 3, 3, spec)
    
    #  Cap vals lower than mean to 0
    spec = np.where(spec < 0, 0, spec)
    
    return spec


# In[17]:


print("Assigning distance scores...")


# In[18]:


spec_dict = dict(zip(labelfile.callID_new.values, labelfile.denoised_spectrograms.values))


# In[19]:


dists=[]

for pair in all_pairs:
    spec = preprocess_spec(spec_dict[pair[0]])
    nb_spec = preprocess_spec(spec_dict[pair[1]])
    dist = calc_mindist(spec, nb_spec)
    dists.append(dist)


# In[20]:


f_nf = pd.DataFrame({'call_a': [x[0] for x in all_pairs],
                    'call_b' : [x[1] for x in all_pairs],
                    'dist_score' : dists})
print("Done")


# # Calculate audio intensity

# In[21]:


# Bandpass filters for calculating audio intensity
LOWCUT = 300.0
HIGHCUT = 3000.0


# In[22]:


# Function that calculates intensity score from 
# amplitude audio data
# Input: 1D numeric numpy array (audio data)
# Output: Float (Intensity)
def calc_audio_intense_score(audio):
    res = 10*math.log((np.mean(audio**2)),10)
    return res

# Butter bandpass filter implementation:
# from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# In[23]:


print("Assigning intensity scores...")

# Using the band-pass filtered signal! 

clean_intensity = []

for call in list(labelfile.callID_new):
    audio = list(labelfile.loc[labelfile.callID_new==call,'raw_audio'])[0]
    sr = list(labelfile.loc[labelfile.callID_new==call,'samplerate_hz'])[0]
    y = butter_bandpass_filter(audio, LOWCUT, HIGHCUT, sr, order=6)
    clean_intensity.append(calc_audio_intense_score(y))

labelfile['intense_score'] = clean_intensity
intense_dict = dict(zip(labelfile.callID_new.values, labelfile.intense_score.values))

f_nf['intense_a'] = [intense_dict[x] for x in f_nf.call_a]
f_nf['intense_b'] = [intense_dict[x] for x in f_nf.call_b]

print("Done")


# # Calculate physical distance

# In[24]:


def dist_2d(x1,y1,x2,y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist


# In[30]:


print("Assigning physical distance scores...")

physical_dist = []

for i in range(f_nf.shape[0]):
    call = f_nf.loc[i,'call_a']    
    x1 = float(labelfile.loc[labelfile.callID_new==call,'x_emitted'])
    y1 = float(labelfile.loc[labelfile.callID_new==call,'y_emitted'])
    
    call = f_nf.loc[i,'call_b'] 
    x2 = float(labelfile.loc[labelfile.callID_new==call,'x_emitted'])
    y2 = float(labelfile.loc[labelfile.callID_new==call,'y_emitted'])
    
    if np.isnan([x1,y1,x2,y2]).any()==True:
        d = np.nan 
    else:
        d = dist_2d(x1,y1,x2,y2)
        
    physical_dist.append(d)
    
f_nf['physical_dist'] = physical_dist
print("Done")


# # Save f_nf file

# In[31]:


f_nf_out = os.path.join(os.path.sep, HOME,'f_nf.csv')
f_nf.to_csv(f_nf_out, sep="\t", index=False)

print("Completed.")

