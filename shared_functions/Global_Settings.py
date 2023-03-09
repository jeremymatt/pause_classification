# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:41:13 2023

@author: jerem
"""
import os

dataset_name = 'test_wav_files'
BERT_labels_fn = 'BERT_training_data.csv'

cd = os.getcwd()
root = os.path.split(cd)[0]

#paths to directories
transcripts_path = os.path.join(root,'data','transcripts',dataset_name)
spectrogram_data_path = os.path.join(root,'data','spectrogram_data',dataset_name)
BERT_results_path = os.path.join(root,'data','BERT_results',dataset_name)
audio_source_path = os.path.join(root,'data','audio_source_files',dataset_name)
random_forest_data_path = os.path.join(root,'data','random_forest_results',dataset_name)

#paths to specific files
BERT_labels_fn = os.path.join(transcripts_path,settings.BERT_labels_fn)

path_list = [
    transcripts_path,
    spectrogram_data_path,
    BERT_results_path,
    audio_source_path,
    random_forest_data_path]

for path in path_list:
    if not os.path.isdir(path):
        os.makedirs(path)