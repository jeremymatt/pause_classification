# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 20:47:43 2022

@author: jerem
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import copy as cp
from matplotlib import rcParams
rcParams.update({'font.size': 15})

import sys
cd = os.getcwd()
root = os.path.split(cd)[0]
shared_functions = os.path.join(root,'shared_functions')
sys.path.insert(0,shared_functions)
import ml_metrics as MLM
import Global_Settings as settings
sys.path.remove(shared_functions)

from sklearn.metrics import confusion_matrix 


#Name of the dataset to process
dataset = settings.dataset_name


convert_codebook_numbers_to_alis_numbers = {
    1:0,
    2:1,
    0:2}


# silence_type_dict = {
#     's0':0,  #non-connectional
#     's1':1,  #Invitational
#     's2':2,  #Emotional
#     's3':2,  #Compassionate
#     's7':'unk'}

"""
#labels for original coding

labels_dict = {
    1:'Invitational',
    2:'Emotional',
    0:'Non-connectional',
    '1':'Invitational',
    '2':'Emotional',
    '0':'Non-connectional',
    'Non-connectional':'Non-connectional',
    'Emotional':'Emotional',
    'Invitational':'Invitational'}
"""

#labels for new coding numbers
labels_dict = {
    6:'Invitational',
    5:'Emotional',
    0:'Other',
    '6':'Invitational',
    '5':'Emotional',
    '0':'Other',
    'Non-connectional':'Non-connectional',
    'Emotional':'Emotional',
    'Invitational':'Invitational'}

run_ID = 'run1'
source = os.path.join(settings.BERT_results_path,run_ID)

labels_path = os.path.join(settings.transcripts_path,settings.settings.BERT_labels_fn)
original_labels_df = pd.read_csv(labels_path)
original_labels_df.set_index('Unnamed: 0',inplace=True,drop=True)

#list the directories for each split in the run
subdirs = [dr for dr in os.listdir(source) if os.path.isdir(os.path.join(source,dr))]

test_data = pd.DataFrame()
train_data = pd.DataFrame()
results = None
ctr = 0
dr = subdirs[0]

#Epoch to use for generating the confusion matrix and calculating metrics
#Epoch 0 is after the first training instance, epoch 1 is after the second 
#training instance, etc
epoch = 1

epochs = range(5)
# epochs = [1]

for epoch in epochs:
    fold_ctr = 0
    
    test_cm = None
    
    cm_order = [
        'Invitational',
        'Emotional',
        'Non-connectional']
    cm_order = [
        'Invitational',
        'Non-connectional']
    
    results = None
    
    for dr in subdirs:
        cur_dir = os.path.join(source,dr)
        
        subsubdirs = [sdr for sdr in os.listdir(cur_dir) if os.path.isdir(os.path.join(cur_dir,sdr))]
        
        sdr = subsubdirs[0]
        for sdr in subsubdirs:
            fold_ctr += 1
            cur_sub_dir = os.path.join(cur_dir,sdr)
            hist_fn = [file for file in os.listdir(cur_sub_dir) if file.startswith('history')][0]
            temp_df = pd.read_csv(os.path.join(cur_sub_dir,hist_fn))
            
            trd = np.array(temp_df.train)
            ted = np.array(temp_df.test)
            
            for i in range(len(trd)):
                test_data.loc[ctr,i] = ted[i]
                train_data.loc[ctr,i] = trd[i]
            ctr += 1
            
            temp_results = pd.read_csv(os.path.join(cur_sub_dir,'results_epoch_{}.csv'.format(epoch)))
            if not type(results) == pd.core.frame.DataFrame:
                results = pd.DataFrame(temp_results)
                # print('making df')
            else:
                # results = results.append(temp_results)
                results = pd.concat([results,temp_results],axis=0)
                # print('appending to df')
                
            
            if os.path.isfile(os.path.join(cur_sub_dir,'test_results_epoch_{}.csv'.format(epoch))):
                
                cur_test_results = pd.read_csv(os.path.join(cur_sub_dir,'test_results_epoch_{}.csv'.format(epoch)))

                temp,true_label_set,pred_label_set = MLM.confusion_matrix(cur_test_results['Final_Code'],cur_test_results['prediction'],labels_dict=labels_dict)
                
                cm_row_order = [key for key in cm_order if key in temp.index]
                cm_col_order = [key for key in cm_order if key in temp.keys()]
                temp = temp.loc[cm_row_order,cm_col_order].values
                    
                if not type(test_cm) == np.ndarray:
                    test_cm = temp
                else:
                    if len(test_cm.shape) == 2:
                        test_cm = np.append(test_cm[:,:,np.newaxis],temp[:,:,np.newaxis],axis=2)
                    else:
                        test_cm = np.append(test_cm,temp[:,:,np.newaxis],axis=2)
        
                        
        results.set_index('eventId',inplace=True,drop=True)
        
        
        for ind in results.index:
            
            if results.loc[ind,'pre silence'] == original_labels_df.loc[ind,'pre silence']:
                results.loc[ind,['Patient','Conversation']] = original_labels_df.loc[ind,['Patient','Conversation']]
            else:
                print("ERROR: context does not match:")
                print('  results dataframe: {}'.format(results.loc[ind,'pre silence']))
                print('   labels dataframe: {}'.format(original_labels_df.loc[ind,'pre silence']))
        
        results.to_csv(os.path.join(cur_dir,'compiled_results_epoch-{}.csv'.format(epoch+1)))
    # for i in range(test_cm.shape[2]):
    #     print(test_cm[:,:,i]) 
    
    
    if os.path.isfile(os.path.join(cur_sub_dir,'test_results_epoch_{}.csv'.format(epoch))):
        test_confusion = pd.DataFrame(test_cm.mean(axis=2),index = cm_order,columns=cm_order) 
        
        test_confusion['n'] = test_cm[:,:,0].sum(axis=1)
        
        accuracy = MLM.calc_accuracy(test_confusion)
        
        precision = MLM.calc_precision(test_confusion)
        
        weighted_precision = MLM.calc_class_weighted_precision(test_confusion)
        
        recall = MLM.calc_recall(test_confusion)
        
        print('\n\nTEST Confusion Matrix:')
        print(test_confusion)
        print('\n Accuracy: {}%\n'.format(100*np.round(accuracy,3)))
        print('{}\n'.format(np.round(precision,3)*100))
        print('{}\n'.format(np.round(weighted_precision,3)*100))
        print('{}'.format(np.round(recall,3)*100))
        
        # print(test_confusion)   
    
        print('\n\n')       
    
    test_mean = test_data.mean(axis=0)
    train_mean = train_data.mean(axis=0)
    x = np.arange(start=0,stop=len(test_mean),step=1)
    
    test_std = test_data.std(axis=0)
    train_std = train_data.std(axis=0)
    num_std = 1.96
    
    test_range = num_std*test_std
    train_range = num_std*train_std
    
    if fold_ctr == 10:
        t_alpha = 2.262 #10-fold
    else:
        t_alpha = 1.990 #100-fold
    
    test_range = t_alpha*test_std/np.sqrt(fold_ctr-1)
    train_range = t_alpha*train_std/np.sqrt(fold_ctr-1)
    
    
    plt.figure(figsize=(12,8))
    
    alpha = .5
    plt.plot(x,test_mean,c='m',label = 'Validation')
    plt.plot(x,train_mean,color='b',label='Train')
    plt.fill_between(x,test_mean-test_range,test_mean+test_range,color='m',alpha = alpha, label = 'Validation 95% CI')
    plt.fill_between(x,train_mean-train_range,train_mean+train_range,color='b',alpha = alpha, label = 'Train 95% CI')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    confusion,true_label_set,pred_label_set = MLM.confusion_matrix(results['label True'],results['label_pred'],labels_dict=labels_dict)
    
    n_other = 12829
    n_emotional = 299
    
    actual_n = [12829,299,226]
    indices = ['Other','Emotional','Invitational']
    
    
    unweighted_confusion = cp.deepcopy(confusion)
    for ctr,ind in enumerate(indices):
        unweighted_confusion.loc[ind,:] = unweighted_confusion.loc[ind,:]*actual_n[ctr]/unweighted_confusion.loc[ind,'n']
    
    unweighted_confusion = np.round(unweighted_confusion,0).astype(int)
    
    accuracy = MLM.calc_accuracy_old(confusion)
    
    precision = MLM.calc_precision_old(confusion)
    
    weighted_precision = MLM.calc_class_weighted_precision_old(confusion)
    
    recall = MLM.calc_recall_old(confusion)
    
    class_weight = False
    stats_list = 'all'
    report_values = False
    stats_df, tfpn_df = MLM.build_stats_df(confusion,stats_list,report_values,class_weight)
    stats_df_unweighted, tfpn_df_unweighted = MLM.build_stats_df(unweighted_confusion,stats_list,report_values,class_weight)
    
    with open('results.txt','w') as f:
        f.write('\n\nEPOCH {}\nBALANCED\nConfusion Matrix:\n'.format(epoch+1))
        f.write(confusion.to_string(header=True,index=True))
        f.write('\n\n')
        f.write('Overall Accuracy: {}%'.format(np.round(accuracy*100,1)))
        f.write('\n\nPer-class TP, TN, FP, FN values:\n')
        f.write('{}\n'.format(tfpn_df.astype(int).to_string(header=True,index=True)))
        f.write('\n\nClass-specific stats:\n')
        f.write('{}\n'.format(stats_df.to_string(header=True,index=True)))
        
        accuracy = MLM.calc_accuracy_old(unweighted_confusion)
        f.write('\n\nEPOCH {}\nUNBALANCED\nConfusion Matrix:\n'.format(epoch+1))
        f.write(unweighted_confusion.to_string(header=True,index=True))
        f.write('\n\n')
        f.write('Overall Accuracy: {}%'.format(np.round(accuracy*100,1)))
        f.write('\n\nPer-class TP, TN, FP, FN values:\n')
        f.write('{}\n'.format(tfpn_df_unweighted.astype(int).to_string(header=True,index=True)))
        f.write('\n\nClass-specific stats:\n')
        f.write('{}\n'.format(stats_df_unweighted.to_string(header=True,index=True)))
        
    print('\n\nEPOCH {}\nConfusion Matrix:'.format(epoch+1))
    print(confusion)
    print('\nOverall Accuracy: {}%\n\nClass-specific stats:'.format(np.round(accuracy*100,1)))
    
    
    print('{}\n'.format(stats_df))
    # print('{}\n'.format(np.round(weighted_precision,3)*100))
    # print('{}'.format(np.round(recall,3)*100))
    # print('{}'.format(np.round(precision,3)*100))
