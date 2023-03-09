# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 07:40:19 2020

@author: jmatt
"""

import os
import sys
cd  = os.getcwd()
shared_functions = os.path.join(os.path.split(cd)[0],'shared_functions')
sys.path.insert(0,shared_functions)

import Global_Settings as settings
import DATA_ORGANIZER
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.models import load_model
import ML_functions as MLF
import numpy as np
import os
import output_aggregator_csv as OA
import pandas as pd
# import platform
import tqdm
from runpy import run_path
import matplotlib
import gen_from_audio as GFA
import copy as cp

import time


sys.path.remove(shared_functions)

matplotlib.interactive(False)


run_full_conv = True

conv_threshold = 0.45
rf_threshold = 0.425

start = time.time()



cd = os.getcwd()

# testing = True
# if testing:
#     algorithm,train_classes,run_ID = r'driver_custom-square-filter_train-from-csv\binary\test_on_clean'.split('\\')
#     weights_file = 'weights_ITR-0_EP-07_ACC-0.9189.hdf5'
#     split = 'human_val'
# else:
#     algorithm,train_classes,run_ID = r'driver_custom-square-filter_train-from-csv\binary\2021-03-30_16-52-30'.split('\\')
#     weights_file = 'weights_EP-03_ACC-0.4651.hdf5'
#     weights_file = 'weights_ITR-0_EP-06_ACC-0.9194.hdf5'
#     # weights_file = 'all'
    
#     split = 'DUP_human_val'
     
algorithm,train_classes,run_ID = r'driver_custom-square-filter_train-from-csv\binary\2022-01-12_19-48-53'.split('\\')

output_base_dir = os.path.join(cd,'output',algorithm,train_classes,run_ID)


# t = run_path(os.path.join(output_base_dir,'driver_settings_csv_dataset.py'))

os.chdir(output_base_dir)
from driver_settings_csv_dataset import *
os.chdir(cd)

print("Settings summary:")
print('  Normalization type: {}'.format(norm_type))
print('  Prediction length: {}-intervals'.format(prediction_len))
print('  Padding: {}'.format(padding))
print('  # times each col is duplicated: {}'.format(dup_cols))

#Location of the training dataset
data_dir = os.path.join(cd,'training_datasets','spectral_data_complete')

#Location of RF results:
rf_results_dir = os.path.join(cd,'RandomForest','output','PCCRI')

#The name of the csv file containing the labels.
label_csv = 'img_labels.csv'

label_df = None
tv_split_df = None

fn = os.path.join(cd,'clip_lengths_storylistening.csv')
clip_len_df = pd.read_csv(fn)

clip_len_df['orig_source_filenames'] = clip_len_df.source_name

for ind in clip_len_df.index:
    clip_len_df.loc[ind,'source_name'] = '{}.csv'.format(str(clip_len_df.loc[ind,'patient_ID']).zfill(4))

fn = os.path.join(data_dir,'min_max_conv.csv')
    
data_range_df = pd.read_csv(fn)


# predict_ids = list(clip_len_df.patient_ID)

# fn = tv_split_df.source_filename[0]
# tv_split_df.loc[tv_split_df.source_filename == fn,'TV_split'].values[0]
# train_on = ['train','val','DUP_human_val']
# train_files = [fn for fn in tv_split_df.source_filename if tv_split_df.loc[tv_split_df.source_filename == fn,'TV_split'].values[0] in train_on]
# for file in train_files:
#     tv_split_df.loc[tv_split_df.source_filename == file,'TV_split'] = 'train'
    
# tv_split_df_bkup = cp.deepcopy(tv_split_df)

# dirs_to_skip = [
#     '__pycache__']

# dirs = [item for item in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir,item)) & (item not in dirs_to_skip)]



# dr = dirs[0]   
# for dr in dirs:

file_list = [f for f in os.listdir(os.path.join(output_base_dir)) if os.path.isfile(os.path.join(output_base_dir,f))]  
weight_list = [f for f in file_list if f.split('.')[-1] == 'hdf5']

# tv_split_df = cp.deepcopy(tv_split_df_bkup)

# fn = '{}.wav'.format(dr)

# tv_split_df.loc[tv_split_df.source_filename == fn,'TV_split'] = 'val'

settings = ('conversation',1,'na')
split = 'val'

test_df = GFA.gen_sample_df(label_df,tv_split_df,clip_len_df,output_base_dir,window_len,label_format,split,settings)

files = set(test_df['filename'])




# if weights_file == 'all':
#     file_list = [f for f in os.listdir(output_base_dir_dir) if len(f.split('.')) >1]
#     weight_list = [f for f in file_list if f.split('.')[-1] == 'hdf5']
# else:
#     if type(weights_file) == list:
#         weight_list = weights_file
#     else:
#         weight_list = [weights_file]


epochs_to_process = [5]
# episodes_to_process = list(range(15))
weights_file = weight_list[4]
# weight_list = [weight_list[0]]
for weights_file in weight_list:
    itr = weights_file.split('_')[1].split('-')[1]
    epoch = weights_file.split('_')[2].split('-')[1]
    output_dir = os.path.join(output_base_dir,'PCCRI-complete_results_ITR-{}_EP-{}'.format(itr,epoch))
    
    print(output_dir)
    
    
    
    if int(epoch) in epochs_to_process:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            
            
        #Load the model from the weights file
        print('Loading the model')
        model = load_model(os.path.join(output_base_dir,weights_file))
        print('Done loading the model')
        
        patient_ID = clip_len_df.patient_ID[0]
        for ctr,patient_ID in enumerate(clip_len_df.patient_ID):
            # print('processing ID:{}'.format(file))
            
            
            
            part_number = int(str(patient_ID)[0])
            subject_number = int(str(patient_ID)[1:])
            
            desc = '{}/{} (subject {}, part {})'.format(str(ctr).zfill(2),len(clip_len_df.patient_ID),str(subject_number).zfill(2),part_number)
            
            file = clip_len_df.loc[clip_len_df.patient_ID == patient_ID,'source_name'].values[0]
            file = '{}.csv'.format(str(patient_ID).zfill(4))
            b=1
            test_df_temp = test_df[test_df['filename'] == file]
            aggregator = OA.OUTPUT_AGGREGATOR(label_df, clip_len_df, file, window_len,patient_id = patient_ID)
            
            
            label_keys = [key for key in test_df.keys() if key[:5] == 'label']
            # norm_type = 'conv'
            batch_size = 10
            shuffle = False
            # dup_cols = None
            train_generator = GFA.GENERATOR(test_df_temp, data_dir, data_range_df, norm_type,shuffle,dup_cols,label_keys,split)
            i=0
            for sample_ind in tqdm.tqdm(test_df_temp.index,total=len(test_df_temp.index),desc=desc):
            # for i,sample_ind in enumerate(test_df_temp.index[:20]):
                # print('ind:{}'.format(i))
                data,label,rng = train_generator.get_sample(sample_ind)
                data = data.reshape([1,data.shape[0],data.shape[1],1])
                pred = model.predict(data)[0]
                if len(pred) == 1:
                    pred = pred[0]
                col = i%(window_len+1)
                i+=1
                # print(col)
                aggregator.append_output(rng,pred,col)
            aggregator.calc_max(threshold=0.45)
            aggregator.to_csv(output_dir)
                

delta = time.time() - start

print('delta: {} seconds'.format(delta))    
            