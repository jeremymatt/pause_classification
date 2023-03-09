# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 09:47:11 2022

@author: jerem
"""

import numpy as np
import pandas as pd
import copy as cp

def confusion_matrix(labels_true,labels_pred,labels_dict=None):
    """
    Generates confusion matrix from true and predicted labels.  Optionally
    renames classes based on labels_dict.  True values are on rows and predicted
    values are on columns

    Parameters
    ----------
    label_true : TYPE
        DESCRIPTION.
    label_pred : TYPE
        DESCRIPTION.
    labels_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    label_true : TYPE pandas dataframe
        DESCRIPTION.

    """
    
    confusion = pd.DataFrame()
    
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    true_label_set = set(labels_true)
    pred_label_set = set(labels_pred)
    all_labels_set = true_label_set.union(pred_label_set)
    
    all_labels_set = list(all_labels_set)
    all_labels_set.sort()
    
    true_label_set = list(true_label_set)
    true_label_set.sort()
    
    pred_label_set = list(pred_label_set)
    pred_label_set.sort()
    
    
    if not type(labels_dict) == dict:
        labels_dict = {}
        for label in all_labels_set:
            labels_dict[label] = str(label)
    
    for true_label in true_label_set:
        true_index = labels_dict[true_label]
        true_mask = labels_true == true_label
        for pred_label in pred_label_set:
            pred_index = labels_dict[pred_label]
            pred_mask = labels_pred == pred_label
            confusion.loc[true_index,pred_index] = int(sum(true_mask&pred_mask))
            
    confusion = confusion.astype(int)
    
    confusion['n'] = confusion.sum(axis=1)
    
    return confusion,true_label_set,pred_label_set


def get_tfpn_df(confusion,class_weight):
    labels = list(confusion.index)
    
    if class_weight:
        for label in labels:
            confusion[label] /= confusion['n']
    
    tfpn_df = pd.DataFrame()
    
    for label in labels:
        not_label = set(labels)
        not_label.remove(label)
        not_label = sorted(list(not_label))
        tfpn_df.loc[label,'tp'] = confusion.loc[label,label]
        tfpn_df.loc[label,'tn'] = confusion.loc[not_label,not_label].sum().sum()
        tfpn_df.loc[label,'fn'] = confusion.loc[label,not_label].sum().sum()
        tfpn_df.loc[label,'fp'] = confusion.loc[not_label,label].sum().sum()
    
    return tfpn_df


def calc_precision(tfpn_df,report_values = True,class_weight = False):
    
    stat = pd.DataFrame()
    
    name = 'PPV(precision)'
    if class_weight:
        name = '{}_CW'.format(name)
    
    for ind in tfpn_df.index:
        numerator = tfpn_df.loc[ind,'tp']
        denominator = tfpn_df.loc[ind,['tp','fp']].sum().sum()
        stat = calc_stat_for_table(stat,ind,name,numerator,denominator,report_values)
        
    return stat


def calc_NPV(tfpn_df,report_values = True,class_weight = False):
    
    stat = pd.DataFrame()
    
    name = 'NPV'
    if class_weight:
        name = '{}_CW'.format(name)
    
    for ind in tfpn_df.index:
        numerator = tfpn_df.loc[ind,'tn']
        denominator = tfpn_df.loc[ind,['tn','fn']].sum().sum()
        stat = calc_stat_for_table(stat,ind,name,numerator,denominator,report_values)
        
    return stat
        
        
def calc_sensitivity(tfpn_df,report_values = True):
    
    stat = pd.DataFrame()
    name = 'sensitivity(recall)'
    
    for ind in tfpn_df.index:
        numerator = tfpn_df.loc[ind,'tp']
        denominator = tfpn_df.loc[ind,['tp','fn']].sum().sum()
        stat = calc_stat_for_table(stat,ind,name,numerator,denominator,report_values)
        
    return stat

def calc_stat_for_table(stat,ind,name,numerator,denominator,report_values):
    
    if report_values:
        stat.loc[ind,name] = numerator/denominator
    else:
        string = '{}% ({}/{})'.format(round(100*numerator/denominator,1),int(numerator),int(denominator))
        stat.loc[ind,name] = string
    
    return stat
        
        
def calc_specificity(tfpn_df,report_values = True,class_weight = False):
    
    stat = pd.DataFrame()
    
    name = 'specificity'
    if class_weight:
        name = '{}_CW'.format(name)
    
    for ind in tfpn_df.index:
        numerator = tfpn_df.loc[ind,'tn']
        denominator = tfpn_df.loc[ind,['tn','fp']].sum().sum()
        stat = calc_stat_for_table(stat,ind,name,numerator,denominator,report_values)
        
    return stat
        
def calc_class_accuracy(tfpn_df,report_values = True,class_weight = False):
    
    stat = pd.DataFrame()
    
    name = 'accuracy'
    if class_weight:
        name = '{}_CW'.format(name)
    
    for ind in tfpn_df.index:
        numerator = tfpn_df.loc[ind,['tp','tn']].sum().sum()
        denominator = tfpn_df.loc[ind,['tp','fn','tn','fp']].sum().sum()
        stat = calc_stat_for_table(stat,ind,name,numerator,denominator,report_values)
        
    return stat

def build_stats_df(confusion,stats_list,report_values,class_weight):
    if stats_list == 'all':
        stats_list = ['accuracy','precision','NPV','sensitivity','specificity']
        
    stats_df = None
    
    tfpn_df = get_tfpn_df(confusion,class_weight)
    
    for stat in stats_list:
        if stat == 'accuracy':
            cur_stat_df = calc_class_accuracy(tfpn_df,report_values,class_weight)
        
        if stat == 'precision':
            cur_stat_df = calc_precision(tfpn_df,report_values,class_weight)

        if stat == 'NPV':
            cur_stat_df = calc_NPV(tfpn_df,report_values,class_weight)
        
        if stat == 'sensitivity':
            cur_stat_df = calc_sensitivity(tfpn_df,report_values)

        if stat == 'specificity':
            cur_stat_df = calc_specificity(tfpn_df,report_values,class_weight)
            
        breakhere=1
        if type(stats_df) == type(None):
            stats_df = cur_stat_df
        else:
            stats_df = pd.concat([stats_df,cur_stat_df],axis=1)
    if report_values:     
        stats_df = np.round(stats_df*100,1)
        
    return stats_df, tfpn_df
    


def calc_accuracy_old(confusion):
    labels = list(confusion.index)
    
    num = 0
    for label in labels:
        num += confusion.loc[label,label]
        
    return num/confusion.loc[labels,labels].sum().sum()

def calc_recall_old(confusion):
    labels = list(confusion.index)
    temp = confusion[labels]
    recall = pd.DataFrame()
    
    for label in labels:
        recall.loc[label,'recall'] = temp.loc[label,label]/temp.loc[label,:].sum()
        
    return recall
    


def calc_precision_old(confusion):
    labels = list(confusion.index)
    
    precision = pd.DataFrame()
    
    for label in labels:
        precision.loc[label,'precision'] = confusion.loc[label,label]/confusion.loc[:,label].sum()
        
    return precision
    
    

def calc_class_weighted_precision_old(confusion):
    working_confusion = cp.deepcopy(confusion)
    
    labels = list(working_confusion.index)
    
    precision = pd.DataFrame()
    
    label = labels[0]
    
    for label in labels:
        
        working_confusion.loc[label,labels] = working_confusion.loc[label,labels]/working_confusion.loc[label,'n']
        
    
    for label in labels:
        precision.loc[label,'class-weighted precision'] = working_confusion.loc[label,label]/working_confusion.loc[:,label].sum()
        
    return precision
    
    
