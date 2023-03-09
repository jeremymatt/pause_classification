# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:51:26 2021

@author: jmatt
"""

import os 
import sys
cd  = os.getcwd()
shared_functions = os.path.join(cd,'shared_functions')
sys.path.insert(0,shared_functions)

import Global_Settings as settings
import pandas as pd
import numpy as np
import extract_spectral_data as ESD
import tqdm
import shutil
import librosa
import time

sys.path.remove(shared_functions)


#Name of the dataset to process
source_audio = settings.dataset_name

#Data path to audio files
source_dir = settings.audio_source_path

#Folder within the output directory to save the file (and therefore script settings)
#used to generate the spectrogram data
gen_file_dir = 'generating_files'
#Name of a csv file containing the lengths of each input audio file
lengths = 'clip_lengths.csv'
file_len = 0.5 #seconds per csv file

#Construct the path to the output directory
output_dir = settings.spectrogram_data_path

#Path where generating files will be saved
gen_file_dir = os.path.join(output_dir,gen_file_dir)

#create the folder if it doesn't exist
if not os.path.isdir(gen_file_dir):
    os.makedirs(gen_file_dir)
    
    
    
try:
    #The list of generating files to archive into the run output directory
    #preserves the settings used to generate a specific dataset
    files = [
        os.path.basename(__file__),
        'extract_spectral_data.py']
    
    for fn in files:
        #build the full path & filename of the source file
        from_file = os.path.join(cd,fn)
        #build the full path & filename of the destination file
        to_file = os.path.join(gen_file_dir,fn)
        #Copy the file
        shutil.copy(from_file,to_file)
    
except:
    print('not running from file')
    
    
#remove any double spaces in the file names
files = [file.replace('  ',' ') for file in os.listdir(source_dir) if file[-4:] == '.wav']

#Find the lengths of each audio file and save to csv in the spectrogram output 
#directory
clip_lengths = ESD.get_clip_lengths(source_dir, files)
clip_lengths.to_csv(os.path.join(output_dir,'clip_lengths.csv'))

#extract the subject numbers from the audio file names (will need to be updated
#for different datasets)

# subject_numbers = [item.split('Sub')[1].split(' ')[1] for item in files]
subject_numbers = [item.split('.')[0] for item in files]
subject_numbers = [str(int(item)).zfill(4) for item in subject_numbers]

#Make a dict mapping patient ID numbers to filenames
temp_dict = {}
for fn,num in list(zip(files,subject_numbers)):
    if num in temp_dict.keys():
        temp_dict[num].append(fn)
    else:
        temp_dict[num] = [fn]

#Make a dict mapping filenames to strings of patient ID numbers
fn_dict = {}
for num in temp_dict.keys():
    ctr = 1
    for fn in temp_dict[num]:
        fn_dict[fn] = '{}'.format(num)
        ctr += 1

#FIle to save the min/max intensity values for the entire dataset
min_max_fn = os.path.join(output_dir,'dataset_minmax.csv')
#If there's an existing file, read it (for restarting)
if os.path.isfile(min_max_fn):
    mm = pd.read_csv(os.path.join(output_dir,'dataset_minmax.csv'))
    dataset_min = mm.loc[0,'min']
    dataset_max = mm.loc[0,'max']
else:
    dataset_min = np.nan
    dataset_max = np.nan
    
#dataframe to store the min/max intensity for each conversation
min_max_conv_df = pd.DataFrame()
start_time = time.time()
total_len = 0
for ctr,file in enumerate(files):
    
    #Init variables to hold dataset min/max intensity values
    file_min = np.nan
    file_max = np.nan
    
    #Path to the source audio file
    source_file = os.path.join(source_dir,file)
    
    #Path to the output directory for the current audio file and make if doesn't
    #exist
    file_dir = os.path.join(output_dir,fn_dict[file])
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    
    #Extract the length of the current audiofile
    clip_len = clip_lengths.loc[clip_lengths['source_name'] == file,'length(s)'].values[0]
    #Track the total length of all input audio files
    total_len += clip_len
    
    #Points at which to break the output into separate .csv files of spectral
    #data
    file_breaks = list(np.array(range(0,int(clip_len*2),int(file_len*2)))/2)    
    file_breaks.append(clip_len)
    
    #TQDM label to track progress fo rthe user
    label = '{}/{}'.format(ctr,len(files))
    
    #generate a list of start/end ranges over which to extract the spectral data
    break_tpls = list(zip(file_breaks[:-1],file_breaks[1:]))
    
    #For each section of audio, extract spectral data and save to file
    for start,stop in tqdm.tqdm(break_tpls,total=len(break_tpls),desc=label):
        #Build the output filename for the current audio section
        output_fn = os.path.join(file_dir,'{}-{}.csv'.format(int(start*2),int(stop*2)))
        try:
            frame_min,frame_max = ESD.extract_spectral_data(source_file,start,stop,output_fn)
        except:
            print('\nFAILURE 1\n')
            try:
                frame_min,frame_max = ESD.extract_spectral_data(source_file,start,stop,output_fn)
            except:
                print('\nFAILURE 2\n')
                frame_min,frame_max = ESD.extract_spectral_data(source_file,start,stop,output_fn)
        #Update min/max values
        dataset_min = np.nanmin([dataset_min,frame_min])
        dataset_max = np.nanmax([dataset_max,frame_max])
        file_min = np.nanmin([file_min,frame_min])
        file_max = np.nanmax([file_max,frame_max])
    
    #Update the conversation level min/max data
    min_max_conv_df.loc[ctr,'filename'] = '{}.csv'.format(fn_dict[file])
    min_max_conv_df.loc[ctr,'min'] = file_min
    min_max_conv_df.loc[ctr,'max'] = file_max
        
    
#store the dataset-level min/max values       
min_max = pd.DataFrame({'min':[dataset_min],'max':[dataset_max]})
min_max.to_csv(os.path.join(output_dir,'dataset_minmax.csv'))
#Store the conversation level min/max data
min_max_conv_df.to_csv(os.path.join(output_dir,'min_max_conv.csv'))


#Print timing information
end_time = time.time()

elapsed_time =end_time-start_time  

print('\nTotal len of files processed: {}secs (or {} mins or {} hours)'.format(total_len,total_len/60, total_len/3600))

print('elapsed time: {}sec (or {}min or {}hrs)'.format(elapsed_time,elapsed_time/60,elapsed_time/3600))

print('{}mins to process 1hr of audio'.format((elapsed_time/60)/(total_len/3600)))
