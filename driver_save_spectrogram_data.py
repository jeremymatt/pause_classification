# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:51:26 2021

@author: jmatt
"""

import os 
import pandas as pd
import numpy as np
import extract_spectral_data as ESD
import tqdm
import shutil
import librosa
import time



cd = os.getcwd()


source_audio = 'test_wav_files'

source_dir = os.path.join(cd,'audio_source_files',source_audio)

gen_file_dir = 'generating_files'
lengths = 'clip_lengths.csv'
file_len = 0.5 #seconds per csv file

# frame_len = 12 #seconds
# if (int(60/frame_len)-60/frame_len) != 0:
#     print("ERROR: 60 sec/min not evenly divisible by frame length")

# frames_per_min = 60/frame_len

output_dir = os.path.join(cd,'spectrogram_data',source_audio)

    
gen_file_dir = os.path.join(output_dir,gen_file_dir)

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
    
    

files = [file.replace('  ',' ') for file in os.listdir(source_dir) if file[-4:] == '.wav']


clip_lengths = ESD.get_clip_lengths(source_dir, files)
clip_lengths.to_csv(os.path.join(output_dir,'clip_lengths.csv'))

# source_dir = os.path.join(cd,source_dir)
# clip_lengths = pd.read_csv(lengths)

# subject_numbers = [item.split('Sub')[1].split(' ')[1] for item in files]
subject_numbers = [item.split('.')[0] for item in files]
subject_numbers = [str(int(item)).zfill(4) for item in subject_numbers]

temp_dict = {}

#%%

for fn,num in list(zip(files,subject_numbers)):
    if num in temp_dict.keys():
        temp_dict[num].append(fn)
    else:
        temp_dict[num] = [fn]
        
fn_dict = {}
for num in temp_dict.keys():
    ctr = 1
    for fn in temp_dict[num]:
        fn_dict[fn] = '{}'.format(num)
        ctr += 1

# fn_list = list(fn_dict.keys())
# fn = fn_list[0]
# id_list = []
# length_list = []
# for fn in fn_list:
#     source_file = os.path.join(source_dir,fn)
#     length = int(np.floor(librosa.get_duration(filename = source_file)))
    
#     id_list.append(fn_dict[fn])
#     length_list.append(length)
    
    
#%%
# clip_lengths = pd.DataFrame({'patient_ID':id_list,'source_name':fn_list,'length(s)':length_list})    
# clip_lengths.to_csv('clip_lengths_storylistening.csv')
#%%

min_max_fn = os.path.join(output_dir,'dataset_minmax.csv')
if os.path.isfile(min_max_fn):
    mm = pd.read_csv(os.path.join(output_dir,'dataset_minmax.csv'))
    dataset_min = mm.loc[0,'min']
    dataset_max = mm.loc[0,'max']
else:
    dataset_min = np.nan
    dataset_max = np.nan
# for file in tqdm.tqdm(files,total=len(files)):
    
min_max_conv_df = pd.DataFrame()
start_time = time.time()
total_len = 0
for ctr,file in enumerate(files):
    
    file_min = np.nan
    file_max = np.nan
    
    # print('Processing file {}/{}'.format(ctr,len(files)))
    source_file = os.path.join(source_dir,file)
    
    file_dir = os.path.join(output_dir,fn_dict[file])
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    
    clip_len = clip_lengths.loc[clip_lengths['source_name'] == file,'length(s)'].values[0]
    total_len += clip_len
    
    file_breaks = list(np.array(range(0,int(clip_len*2),int(file_len*2)))/2)
    frame_breaks = list(range(0,clip_len,20))
    
    file_breaks.append(clip_len)
    
    label = '{}/{}'.format(ctr,len(files))
    
    break_tpls = list(zip(file_breaks[:-1],file_breaks[1:]))
    
    for start,stop in tqdm.tqdm(break_tpls,total=len(break_tpls),desc=label):
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
        # print('min:{},max:{}'.format(frame_min,frame_max))
        dataset_min = np.nanmin([dataset_min,frame_min])
        dataset_max = np.nanmax([dataset_max,frame_max])
        file_min = np.nanmin([file_min,frame_min])
        file_max = np.nanmax([file_max,frame_max])
        
    min_max_conv_df.loc[ctr,'filename'] = '{}.csv'.format(fn_dict[file])
    min_max_conv_df.loc[ctr,'min'] = file_min
    min_max_conv_df.loc[ctr,'max'] = file_max
        
    
        
min_max = pd.DataFrame({'min':[dataset_min],'max':[dataset_max]})
min_max.to_csv(os.path.join(output_dir,'dataset_minmax.csv'))

min_max_conv_df.to_csv(os.path.join(output_dir,'min_max_conv.csv'))



end_time = time.time()

elapsed_time =end_time-start_time  

print('\nTotal len of files processed: {}secs (or {} mins or {} hours)'.format(total_len,total_len/60, total_len/3600))

print('elapsed time: {}sec (or {}min or {}hrs)'.format(elapsed_time,elapsed_time/60,elapsed_time/3600))

print('{}mins to process 1hr of audio'.format((elapsed_time/60)/(total_len/3600)))
