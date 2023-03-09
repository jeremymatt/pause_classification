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
trained_cnn_model_path = os.path.join(root,'data','CNN_results','trained_model')
cnn_predictions_only_path = os.path.join(root,'data','cnn_predictions_only',dataset_name)
cnn_predictions_with_labels_path = os.path.join(root,'data','cnn_predictions_with_labels',dataset_name)

#paths to specific files
BERT_labels_fn = os.path.join(transcripts_path,settings.BERT_labels_fn)
trained_cnn_model_fn = os.path.join(trained_cnn_model_path,'weights_ITR-0_EP-05_ACC-0.9158.hdf5')

path_list = [
    transcripts_path,
    spectrogram_data_path,
    BERT_results_path,
    audio_source_path,
    random_forest_data_path,
    cnn_predictions_only_path,
    cnn_predictions_with_labels_path]

for path in path_list:
    if not os.path.isdir(path):
        os.makedirs(path)