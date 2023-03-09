# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:24:56 2020

@author: jmatt
"""

import pandas as pd
import numpy as np
import os

class OUTPUT_AGGREGATOR:
    """
    Object to aggregate results from different models
    """
    def __init__(self,label_df,clip_len_df,filename,window_len,patient_id = None):
        self.filename = filename
        if patient_id == None:
            self.patient_id = filename.split('.')[0]
        else:
            self.patient_id = patient_id
        self.window_len = window_len+1
        self.clip_len = 2*clip_len_df.loc[clip_len_df['source_name'] == filename,'length(s)'].values[0]
        if type(label_df) == pd.core.frame.DataFrame:
            label_df = label_df[label_df['filename'] == filename]
            pause_tpls = list(zip(label_df['silence_start_s'],label_df['silence_end_s']))
            true_sum = np.zeros(self.clip_len)
            for start,end in pause_tpls:
                true_sum[int(2*start):int(2*end)] = 1
                
        else:
            true_sum = np.zeros(self.clip_len)*np.nan
            
            
        time = np.array(range(self.clip_len))/2
        self.results_df = pd.DataFrame({'time':time,'true_sum':true_sum})
        
        nan_array = np.zeros(self.clip_len)
        nan_array[:] = np.nan
        
        for i in range(self.window_len):
            col_name = 'result_{}'.format(i)
            self.results_df[col_name] = nan_array
            
    def append_output(self,rng,results,col):
        start,end = rng
        start = int(start*2)
        end = int(end*2)
        col_name = 'result_{}'.format(col)
        self.results_df.loc[start:end-1,col_name] = results
        breakhere=1
            
        
    def calc_average(self,threshold=0.5):
        result_cols = ['result_{}'.format(i) for i in range(self.window_len)]
        
        results = np.array(self.results_df[result_cols])
        result_mean = np.nanmean(results,axis=1)
        self.results_df['pred_mean'] = result_mean
        pred_sum = result_mean>threshold
        pred_sum = pred_sum.astype(int)
        self.results_df['pred_sum'] = pred_sum
        
    def calc_max(self,threshold=0.5):
        result_cols = ['result_{}'.format(i) for i in range(self.window_len)]
        
        results = np.array(self.results_df[result_cols])
        result_max = np.nanmax(results,axis=1)
        self.results_df['result_max'] = result_max
        prediction = result_max>threshold
        prediction = prediction.astype(int)
        self.results_df['prediction_max'] = prediction
                
        
        
    def to_csv(self,output_dir):
        
        self.results_df.to_csv(os.path.join(output_dir,'results_{}.csv'.format(self.patient_id)))
        
    def load_RF_results(self,rf_results_dir):
        # fn = os.path.join(rf_results_dir,'{}.xls'.format(aggregator.patient_id))
        fn = os.path.join(rf_results_dir,'{}.xls'.format(self.patient_id))
        rf_data = pd.read_excel(fn)
        
        
        
        