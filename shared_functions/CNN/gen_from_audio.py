# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:23:34 2021

@author: jmatt
"""
import numpy as np
import keras
import pandas as pd
import os
import random

def start_stop_label_tpl(start,window_len,end,before,after,clip_len):
    
    start = int(2*start)
    end = int(2*end)
    clip_len = 2*clip_len
    # label_len = 4
    before = before-window_len
    after = after-window_len
    rng_start = start-before
    rng_end = end+after
    
    starts = np.array(range(rng_start-window_len,rng_end+1))
    breakhere=1
    return starts
    

def make_sample_df(tpl_list,labels):
    
    sample_df = pd.DataFrame(tpl_list,columns = ['filename','split','start','end','pred_start','pred_end'])
    breakhere=1
    if len(labels.shape) == 1:
        sample_df['label_0'] = labels
    else:
        for i in range(labels.shape[1]):
            sample_df['label_{}'.format(i)] = labels[:,i]
    # label_max = sample_df.groupby(['filename','start','end']).label.transform(max)
    # sample_df = sample_df[sample_df.label == label_max]
    sample_df.drop_duplicates(inplace=True)
    sample_df.sort_values(['filename','start'],inplace=True)
    sample_df.reset_index(inplace=True,drop=True)
    
    return sample_df

def gen_sample_df(label_df,tv_split_df,clip_len_df,output_dir,window_len,label_format,split,settings):
    #settings = ('conversation',hop)
    #settings = ('per-pause',before,after)
    
    if settings[0] == 'per-pause':
        if label_format['rand_window_start']:
            _,before_set,after_set = settings
            delta = before_set+after_set
        else:
            _,before,after = settings
    if type(tv_split_df) != pd.core.frame.DataFrame:
        fn_list = list(clip_len_df.source_name)
        
        data_source_fn_list = ["{}.csv".format(str(ID).zfill(4)) for ID in clip_len_df.patient_ID]
    else:
        tv_split_df = tv_split_df[tv_split_df.TV_split == split]
        fn_list = list(tv_split_df.source_filename)
        data_source_fn_list = fn_list
    tpl_list = []
    labels = []
    pad_s,pad_e = label_format['padding']
    for fn,datasource_fn in list(zip(fn_list,data_source_fn_list)):
        clip_len = clip_len_df.loc[clip_len_df.source_name == fn,'length(s)'].values[0]
        
              
        pad_s,pad_e = label_format['padding']

        if settings[0] == 'conversation':
            hop = settings[1]
            starts = np.array(range(0,2*clip_len+hop,hop))
        else:
            
            if type(label_df) != pd.core.frame.DataFrame:
                temp_label_df = pd.DataFrame()
            else:
                temp_label_df = label_df[label_df.filename == fn]
            
            starts = []
            for ind in temp_label_df.index:
                start,end = temp_label_df.loc[ind,['silence_start_s','silence_end_s']]
                
                if label_format['rand_window_start']:
                    before = random.randint(0,delta)
                    after = delta-before
                    # print('before:{}, after:{}'.format(before,after))
                
                starts.extend(start_stop_label_tpl(start,window_len,end,before,after,clip_len))
            starts = np.array(starts)
            
            
        breakhere=1    
        ends = starts+window_len
        mask = ends<=2*clip_len
        starts = starts[mask]/2
        ends = ends[mask]/2
        mask = starts>=0
        starts = starts[mask]
        ends = ends[mask]
        pred_starts = starts+pad_s/2
        pred_ends = ends-pad_e/2
        fn_list = []
        for i in range(len(starts)):
            fn_list.append(datasource_fn)
        split_list = np.array(starts).astype(str)
        split_list[:] = split
        # list(zip(fn_list,starts,ends,pred_starts,pred_ends))
        if type(label_df) == pd.core.frame.DataFrame:
            tmp_labels = make_label_list(label_df,fn,pred_starts,pred_ends,clip_len,label_format)
            labels.extend(tmp_labels)
        
        new_tpls = list(zip(fn_list,split_list,starts,ends,pred_starts,pred_ends))
        
        tpl_list.extend(new_tpls)
    
    # print('tpl list: {}'.format(tpl_list[:5]))        
    if type(label_df)  != pd.core.frame.DataFrame:
        labels = np.zeros(len(tpl_list))
    else:
        labels = np.array(labels)
    if label_format == 'binary':
        labels = labels.flatten()
    labels = labels.astype(int)
    sample_df = make_sample_df(tpl_list,labels)
    if output_dir != None:
        sample_df.to_csv(os.path.join(output_dir,f'labels_new_{split}.csv'))
    return sample_df
    
def make_label_list(label_df,fn,pred_starts,pred_ends,clip_len,label_format):
    label_df_temp = label_df[label_df.filename == fn]
    
    if label_format['type'] == 'binary':
        label_list = []
        for ps,pe in list(zip(pred_starts,pred_ends)):
            m1 = ps >= label_df_temp['silence_start_s']
            m2 = pe <= label_df_temp['silence_end_s']
            label_list.append(np.any(m1&m2).astype(int))
        # list(zip(pred_starts,pred_ends,label_list))
    else:
        master_labels = np.zeros(clip_len*2)
        for ind in label_df_temp.index:
            s_ind = int(2*label_df_temp.loc[ind,'silence_start_s'])
            e_ind = int(2*label_df_temp.loc[ind,'silence_end_s'])
            master_labels[s_ind:e_ind] = 1
        # tpls = list(zip(np.array(range(len(master_labels)))/2,master_labels))
        label_list = []
        for ps,pe in list(zip(pred_starts,pred_ends)):
            label_list.append(master_labels[int(2*ps):int(2*pe)])
        # list(zip(pred_starts,pred_ends,label_list))
        
    return label_list

def make_split_sample(label_df,tv_split_df,before,after,split,output_dir):
    ##########
    #TODO: make sure that samples are w/in clip length
    ##########
    tv_split_df = tv_split_df[tv_split_df.TV_split == split]
    sample_df = pd.DataFrame()
    samp_ind = 0
    for fn in tv_split_df.source_filename:
        label_temp_df = label_df[label_df.filename == fn]
        for ind in label_temp_df.index:
            sample_df.loc[samp_ind,'filename'] = fn
            sample_df.loc[samp_ind,'split'] = split
            sample_df.loc[samp_ind,'start'] = label_temp_df.loc[ind,'silence_start_s']  
            sample_df.loc[samp_ind,'end'] = label_temp_df.loc[ind,'silence_end_s'] 
            sample_df.loc[samp_ind,'pre_start'] = sample_df.loc[samp_ind,'start']-before
            sample_df.loc[samp_ind,'post_end'] = sample_df.loc[samp_ind,'end']+after
            sample_df.loc[samp_ind,'label'] = label_temp_df.loc[ind,'silence_type']
            samp_ind+=1
    
    sample_df.to_csv(os.path.join(output_dir,f'labels_new_{split}.csv'))
    return sample_df
            
            
class GENERATOR:
    def __init__(self,sample_df, data_dir, data_range_df, norm_type,shuffle,dup_cols,label_keys,name):
        self.sample_df = sample_df
        self.data_dir = data_dir
        self.data_range_df = data_range_df
        self.norm_type = norm_type
        self.shuffle = shuffle
        self.dup_cols = dup_cols
        self.label_keys = label_keys
        if self.norm_type == 'dataset':
            self.data_range = (data_range_df['min'].min(),data_range_df['max'].max())
        else:
            self.data_range = norm_type
        self.name = name
        
    def set_batch_size(self,batch_size,print_batch):
        self.batch_size = batch_size
        self.steps = self.sample_df.shape[0] // batch_size
        self.print_batch = print_batch
        
    def breakhere(self):
        breakhere=1

    def generator(self,predict=False):
        #norm_type = one of "dataset", None, "conv", "frame"
        
        btch = 0
        i = 0
        breakhere=1
        while True:
            batch = {'data': [], 'labels': []}
            for b in range(self.batch_size):
                if (i == self.sample_df.shape[0]) and self.shuffle:
                    i = 0
                    self.sample_df = self.sample_df.sample(frac=1).reset_index(drop=True)
                    
                data,label,pred_range = self.get_sample(i)
                
                batch['data'].append(data)
                batch['labels'].append(label)
    
                i += 1
            btch += 1
            batch['data'] = np.array(batch['data'])
            if len(batch['data'].shape) < 4:
                batch['data'] = batch['data'][:,:,:,np.newaxis]
            # Convert labels to categorical values
            batch['labels'] = np.array(batch['labels'])
            if self.print_batch:
                print('{} batch {} (size={})'.format(self.name,btch,batch['data'].shape[0]))
            if len(self.label_keys) == 1:
                batch['labels'] = batch['labels'].flatten()
                
            breakhere=1
            
            if predict:
                yield batch['data']
            else:
                yield batch['data'], batch['labels']
    

    def get_sample(self,i):
        #norm_type = one of "dataset", None, "conv", "frame"
       
                    
        keys = ['filename','start','end']
        # i = test_df_temp.index[0]
        # source_file,start,end = train_generator.sample_df.loc[i,keys]
        
        try:
            source_file,start,end = self.sample_df.loc[i,keys]
        except:
            breakhere=1
        label = list(self.sample_df.loc[i,self.label_keys])
        source_dir = os.path.join(self.data_dir,source_file.split('.')[0])
        # source_dir = os.path.join(train_generator.data_dir,source_file.split('.')[0])
        
        
        start = int(2*start)
        end = int(2*end)
        
        if self.norm_type == 'conv':
            try:
                data_range = (
                    self.data_range_df.loc[self.data_range_df.filename == source_file,'min'].values[0],
                    self.data_range_df.loc[self.data_range_df.filename == source_file,'max'].values[0])
            except:
                breakhere=1
        else:
            data_range = self.data_range
                
            
        data = self.min_max_norm(self.load_x(source_dir,start,end),data_range)
        
        pred_range = (self.sample_df.loc[i,'pred_start'],self.sample_df.loc[i,'pred_end'])
        
        return data,label,pred_range
                
             
    
    def load_x(self,source_dir,start,end):
        frame_list = list(range(start,end+1))
        frame_tpls = list(zip(frame_list[:-1],frame_list[1:]))
        file_list = ["{}-{}.csv".format(tpl[0],tpl[1]) for tpl in frame_tpls]
        fn = os.path.join(source_dir,file_list[0])
        frame = np.loadtxt(open(fn,'rb'),delimiter = ',')
        for file in file_list[1:]:
            fn = os.path.join(source_dir,file)
            frame = np.append(frame,np.loadtxt(open(fn,'rb'),delimiter = ','),axis=1)
            
        if self.dup_cols != None:
            temp = np.zeros([frame.shape[0],frame.shape[1]*self.dup_cols])
            for i in range(frame.shape[1]):
                for ii in range(self.dup_cols):
                    temp[:,i*self.dup_cols+ii] = frame[:,i]
            frame = temp
        return frame
    
    def min_max_norm(self,frame,data_range):
        if data_range == None:
            return frame
        elif data_range == 'frame':
            data_range = (frame.min(),frame.max())
        
        frame = (frame-data_range[0])/(data_range[1]-data_range[0])
        return frame
    
