# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 07:40:19 2020

@author: jmatt
"""

import os
import sys
cd  = os.getcwd()
shared_functions = os.path.join(os.path.split(cd)[0],'shared_functions')
CNN_functions = os.path.join(shared_functions,'CNN')
sys.path.insert(0,shared_functions)  
sys.path.insert(0,CNN_functions)


import Global_Settings as settings
sys.path.insert(0,settings.trained_ccn_model_path)
import driver_settings_csv_dataset as model_settings

from keras.models import load_model
import os
import output_aggregator_csv as OA
import pandas as pd
# import platform
import tqdm
import matplotlib
import gen_from_audio as GFA

import time


sys.path.remove(shared_functions)
sys.path.remove(CNN_functions)
sys.path.remove(settings.trained_ccn_model_path)

matplotlib.interactive(False)


run_full_conv = True

conv_threshold = 0.45
rf_threshold = 0.425

start = time.time()

     
algorithm,train_classes,run_ID = r'driver_custom-square-filter_train-from-csv\binary\2022-01-12_19-48-53'.split('\\')

output_base_dir = os.path.join(cd,'output',algorithm,train_classes,run_ID)

print("Settings summary:")
print('  Normalization type: {}'.format(model_settings.norm_type))
print('  Prediction length: {}-intervals'.format(model_settings.prediction_len))
print('  Padding: {}'.format(model_settings.padding))
print('  # times each col is duplicated: {}'.format(model_settings.dup_cols))

#Location of the training dataset
data_dir = settings.spectrogram_data_path

#Location of RF results:
rf_results_dir = settings.random_forest_data_path


label_df = None
tv_split_df = None

fn = os.path.join(settings.spectrogram_data_path,'clip_lengths.csv')
clip_len_df = pd.read_csv(fn)

clip_len_df['orig_source_filenames'] = clip_len_df.source_name

for ind in clip_len_df.index:
    clip_len_df.loc[ind,'source_name'] = '{}.csv'.format(str(clip_len_df.loc[ind,'patient_ID']).zfill(4))

fn = os.path.join(data_dir,'min_max_conv.csv')
    
data_range_df = pd.read_csv(fn)


prediction_settings = ('conversation',1,'na')
split = 'val'

test_df = GFA.gen_sample_df(label_df,
                            tv_split_df,
                            clip_len_df,
                            output_base_dir,
                            model_settings.window_len,
                            model_settings.label_format,
                            split,
                            prediction_settings)

files = set(test_df['filename'])



weights_file = os.path.split(settings.trained_cnn_model_fn)[1]
itr = weights_file.split('_')[1].split('-')[1]
epoch = weights_file.split('_')[2].split('-')[1]
output_dir = settings.cnn_predictions_only_path

print(output_dir)


    
#Load the model from the weights file
print('Loading the model')
model = load_model(settings.trained_cnn_model_fn)
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
    aggregator = OA.OUTPUT_AGGREGATOR(label_df,
                                      clip_len_df,
                                      file,
                                      model_settings.window_len,
                                      patient_id = patient_ID)
    
    
    label_keys = [key for key in test_df.keys() if key[:5] == 'label']
    # norm_type = 'conv'
    batch_size = 10
    shuffle = False
    # dup_cols = None
    train_generator = GFA.GENERATOR(test_df_temp,
                                    data_dir,
                                    data_range_df,
                                    model_settings.norm_type,
                                    shuffle,
                                    model_settings.dup_cols,
                                    label_keys,split)
    i=0
    for sample_ind in tqdm.tqdm(test_df_temp.index,total=len(test_df_temp.index),desc=desc):
    # for i,sample_ind in enumerate(test_df_temp.index[:20]):
        # print('ind:{}'.format(i))
        data,label,rng = train_generator.get_sample(sample_ind)
        data = data.reshape([1,data.shape[0],data.shape[1],1])
        pred = model.predict(data)[0]
        if len(pred) == 1:
            pred = pred[0]
        col = i%(model_settings.window_len+1)
        i+=1
        # print(col)
        aggregator.append_output(rng,pred,col)
    aggregator.calc_max(threshold=0.45)
    aggregator.to_csv(output_dir)
                

delta = time.time() - start

print('delta: {} seconds'.format(delta))    
            