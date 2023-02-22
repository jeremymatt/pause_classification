# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:00:43 2020

@author: jmatt
"""

import librosa
from librosa.feature import melspectrogram
import numpy as np
import os
import pandas as pd


def get_clip_lengths(source_dir, files):
    clip_lengths = pd.DataFrame()
    
    file = files[0]
    for ctr,file in enumerate(files):
        patient_id = int(file.split('.')[0])
        file_path = os.path.join(source_dir,file)
        
        # sub_sample, f_sample = librosa.load(file)
        
        # length = int(sub_sample.shape[0]/f_sample)
        
        length = int(np.floor(librosa.get_duration(filename = file_path)))
        
        clip_lengths.loc[ctr,'ID'] = patient_id
        clip_lengths.loc[ctr,'source_name'] = file
        clip_lengths.loc[ctr,'length(s)'] = length
    
    clip_lengths[['ID','length(s)']] = clip_lengths[['ID','length(s)']].astype(int)
    
    return clip_lengths
    
  
def extract_spectral_data(source_file,start,stop,output_fn):
    """
    Function to generate a spectrogram and save to file.  Optionally will save
    The associated audio to file as well

    Parameters
    ----------
    inputs : TYPE tuple
        DESCRIPTION.
        Contains:
            1. The complete path of the source file
            2. The complete path of the output spectogram
            3. The complete path of the output clip (if clips are being saved)
            4. The start of the clip in seconds
            5. The end of the clip in seconds
        

    Returns
    -------
    None.

    """
    
    
    #Load the clip from the source file
    [sub_sample, FSample] = librosa.load(source_file,offset=start,duration=stop-start)
    # [sub_sample, FSample] = librosa.load(file,offset=start,duration=end-start)
    # print('Took {}sec to load using librosa'.format(time.time()-then))
       
    
    # sub_sample = samples
    
    
    ###############################################
    #These are the best settings I was able to find
    ###############################################
    
    #Power cutoff threshold.  Frequencies below the threshold power are not
    #displayed
    power = 0.75
    power = 1.25
    #How many samples to skip over when moving the window
    secs = stop-start
    denominator = 300*secs/12-1
    hop_length = int(np.floor(len(sub_sample)/denominator))
    #The maximum and minimum frequencies to include in the output 
    #spectrogram
    max_freq = 4000
    min_freq = 36
    #generate the spectrogram
    t = melspectrogram(y=sub_sample, sr=FSample,
                    n_fft = hop_length*6,
                    hop_length=hop_length,
                    win_length=hop_length*2,
                    window = "hann",
                    center=True,
                    pad_mode='reflect',
                    power=power,
                    n_mels=300,
                    fmin=min_freq,
                    fmax=max_freq)
    
       
    #convert amplitude to decibels with a reference power of 100
    db = librosa.amplitude_to_db(t,ref=100)
    db = np.flip(db,axis=0)
    
    if output_fn == "return":
        return db.min(),db.max(),db
    else:    
        np.savetxt(output_fn,db,delimiter=',')
        
        return db.min(),db.max()
    
    
def check_sample_rates(source_file,start,stop):
    """
    Function to generate a spectrogram and save to file.  Optionally will save
    The associated audio to file as well

    Parameters
    ----------
    inputs : TYPE tuple
        DESCRIPTION.
        Contains:
            1. The complete path of the source file
            2. The complete path of the output spectogram
            3. The complete path of the output clip (if clips are being saved)
            4. The start of the clip in seconds
            5. The end of the clip in seconds
        

    Returns
    -------
    None.

    """
    
    
    #Load the clip from the source file
    [sub_sample, FSample] = librosa.load(source_file,offset=start,duration=stop-start)
    # [sub_sample, FSample] = librosa.load(file,offset=start,duration=end-start)
    # print('Took {}sec to load using librosa'.format(time.time()-then))
    
    return len(sub_sample), FSample
