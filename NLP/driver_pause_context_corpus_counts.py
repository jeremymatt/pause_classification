# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 22:18:51 2022

@author: jerem
"""

import os
import sys
cd  = os.getcwd()
shared_functions = os.path.join(os.path.split(cd)[0],'shared_functions')
sys.path.insert(0,shared_functions)

import Global_Settings as settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nlpFunctions as nlpf
import corpus_word_lists as CWL
import seaborn as sns
import scipy.stats as stats
from matplotlib import rcParams
import setupPCCRIconversations as stup

dataset = settings.dataset_name
#Choose the pause classifications for which to calculate the corpus percentages
epoch_num = 2
data_df_path = os.path.join(os.path.split(cd)[0],'data',dataset,'run1','CV_output-2022-12-11_23-05-39_folds0-10')
#path to the file containing the classifications for each pause and the transcript
#context for each pause
data_df = pd.read_csv(os.path.join(data_df_path,'compiled_results_epoch-{}.csv'.format(epoch_num)))

use_true_labels = False
if use_true_labels:
    #Use the true (human-identified labels)
    label_col = 'label True'
else:
    #Use the ML-predicted pause labels
    label_col = 'label_pred'


#path to the location of the transcripts
transcript_path = os.path.join(os.path.split(cd)[0],'Data','transcripts',dataset)

#set of patient IDs
patients = set(data_df.Patient)
#dataframe containing a list of primary conversations for each patient
primary_convo_df = pd.read_csv('main_convo_data.csv')

#If include fill transcripts, then load the complete primary transcript for
#each patient
include_full_transcripts = True
if include_full_transcripts:
    conversations = stup.PCCRI_setup({},transcripts_path = transcript_path)
    patient_primary_convo_list = [(pid,primary_convo_df.loc[primary_convo_df.p0_patient_id==pid,'primary_convo'].values[0]) for pid in patients]
    full_transcript_text = []
    for pid,cid in patient_primary_convo_list:
        conv = conversations.convs[conversations.getConv_i(pid,cid)]
        text = ' '.join([' '.join(turn) for turn in conv.turns])
        full_transcript_text.append(text)

#find the total number of words in each pre-silence context
num_words = np.array([len(string.split(' ')) for string in data_df['pre silence']])

#Exclude contexts that don't have any words
mask = num_words>=0
data_df = data_df.loc[mask,:]

#update list of word counts
num_words = np.array([len(string.split(' ')) for string in data_df['pre silence']])

#Plot histogram of number of words in each pre-context
# plt.hist(num_words,20)

#Find the set of labels
label_types = set(data_df[label_col])

#Mask non-connectional pauses
non_connectional_mask = data_df[label_col] == 0
num_non_connectional = sum(non_connectional_mask)

#Mask the emotional pauses
emotional_mask = data_df[label_col] == 5
num_emotional = sum(emotional_mask)

#Mask the invitational pauses
invitational_mask = data_df[label_col] == 6
num_invitational = sum(invitational_mask)

#Settings for how to aggregate the contexts
per_patient = True  #True = Aggregate the contexts for each patient into a single block of text
per_type = True #Aggregate all text for each type 
emotional_invitational_separately = True 


if not per_patient:

    non_connectional_text = list(data_df.loc[non_connectional_mask,'pre silence'])
    # non_connectional_text = ' '.join(non_connectional_text)
    
    emotional_text = list(data_df.loc[emotional_mask,'pre silence'])
    # emotional_text = ' '.join(emotional_text)
    
    invitational_text = list(data_df.loc[invitational_mask,'pre silence'])
    # invitational_text = ' '.join(invitational_text)
    
else:
    text_list = []
    
    for mask in [non_connectional_mask,emotional_mask,invitational_mask]:
        temp = data_df.loc[mask,:]
        patients = sorted(list(set(temp.Patient)))
        temp_text_list = []
        for patient in patients:
            patient_df = temp.loc[temp.Patient==patient,:]
            text = ' '.join(patient_df['pre silence'])
            temp_text_list.append(text)
        text_list.append(temp_text_list)
        
    non_connectional_text,emotional_text,invitational_text = text_list

#Build lists of text blocks for subsequent processing (either combining 
#the connectional types or keeping separate)
if emotional_invitational_separately:
    text_list = [
        ('Emotional',emotional_text),
        ('Invitational',invitational_text),
        ('Non-Connectional',non_connectional_text)]
else:
    connectional_text = []
    connectional_text.extend(emotional_text)
    connectional_text.extend(invitational_text)
    text_list = [
        ('Connectional',connectional_text),
        ('Non-Connectional',non_connectional_text)]
        

if include_full_transcripts:
    text_list.append(('Complete transcripts',full_transcript_text))
    

if per_type:
    temp = []
    for typ,text in text_list:
        text = ' '.join(text)
        temp.append((typ,[text]))
        word_count = len(text.split(' '))
        print('{}: {} words'.format(typ,word_count))
    text_list = temp
else:
    fig,ax = plt.subplots(1,len(text_list),figsize = [len(text_list)*10,10])
    ctr = 0
    for typ,text in text_list:
        word_counts = []
        for txt_sample in text:
            word_count = len(txt_sample.split(' '))
            word_counts.append(word_count)
        ax[ctr].hist(word_counts,bins=20)
        if per_patient:
            ax[ctr].set_title('{} by patient'.format(typ))
        else:
            ax[ctr].set_title('{} by pause'.format(typ))
        ax[ctr].set_xlabel('# words')
        ax[ctr].set_ylabel('Count')
        ctr += 1
            
for typ,text in text_list:
    print('{}: {} items in text list'.format(typ,len(text)))
    #%%
        

#Make a list of label/corpus pairs
corpus_list = [
    ('Temporal','temporal_reference'),
    ('Uncertainty',CWL.getStrongConfusionWords()),
    ('Loneliness',CWL.getStrongLonelinessWords()),
    ('Symptoms',CWL.getSymptomWords()),
    ('Treatment',CWL.getTreatmentWords()),
    ('Prognosis',CWL.getTreatmentWords()),
    ('1st person singular',CWL.getFirstPersonSingular()),
    ('1st person plural',CWL.getFirstPersonPlural()),
    ('2nd/3rd person',CWL.get2nd3rdPerson())]



labels = []

results_dict = {}

results_df = pd.DataFrame()


typ,text = text_list[0]
label,corpus = corpus_list[2]


print('Processing Text')

num_text_examples = 0
for typ,txt in text_list:
    num_text_examples += len(txt)

labels = []
for label,corpus in corpus_list:
    text_ctr = 0
    for ctr,tpl in enumerate(text_list):
        typ,cur_text = tpl
        results_dict[typ] = {}
        
        if corpus == 'temporal_reference':
            labels.extend(['Future','Past'])
            results_dict[typ]['Future'] = []
            results_dict[typ]['Past'] = []
            
        else:
            labels.append(label)
            results_dict[typ][label] = []
        
        for text in cur_text:
            print('\r{}, {}: {}/{}                                '.format(typ,label,text_ctr,num_text_examples),end='')
            
            count_dict = nlpf.corpus_word_counts_by_bin(corpus, text, 1)
            num_words = count_dict['total_transcript_words']
            results_df.loc[text_ctr,'num_words'] = num_words
            if corpus == 'temporal_reference':
                future = count_dict[0]['future']
                past = count_dict[0]['past']
                results_dict[typ]['Future'].append(100*future/num_words)
                results_dict[typ]['Past'].append(100*past/num_words)
                results_df.loc[text_ctr,'Future'] = 100*future/num_words
                results_df.loc[text_ctr,'Future_count'] = future
                results_df.loc[text_ctr,'Past'] = 100*past/num_words
                results_df.loc[text_ctr,'Past_count'] = past
                results_df.loc[text_ctr,'group_by'] = typ
            else:
                count = count_dict[0]['total_count']
                results_dict[typ][label].append(100*count/num_words)
                results_df.loc[text_ctr,label] = 100*count/num_words
                results_df.loc[text_ctr,'{}_count'.format(label)] = count
                results_df.loc[text_ctr,'group_by'] = typ
                
            text_ctr += 1

#%%

# types = [tpl[0] for tpl in corpus_list]
# if 'Temporal' in types:
#     boxplot_keys = ['Future','Past']
#     boxplot_keys.extend(types)
#     boxplot_keys.remove('Temporal')
# else:
#     boxplot_keys = types
 
   
    
count_keys = [key for key in results_df.keys() if key.endswith('count')]
boxplot_keys = [key.split('_')[0] for key in count_keys]

print('per_type is {}'.format(per_type))
if per_type:
    print('  pertype = true case')
    
    rcParams.update({'font.size': 25})
    x = np.arange(len(boxplot_keys))
    width = .75
    if len(text_list) == 3:
        shift_list = [-1,0,1.25]
    else:
        shift_list = [-1.5,-0.5,0.5,1.75]
    
    
    
    #[1] McGill, R., J. W. Tukey, and W. A. Larsen. "Variations of Boxplots." The American Statistician. Vol. 32, No. 1, 1978, pp. 12–16.
    
    fig,ax = plt.subplots(figsize = [20,15])
    
    rects = []
    
    ctr = 0
    standard_errors = None
    [item[0] for item in text_list]
    
    color_dict = {}
    color_dict['Non-Connectional'] = 'lightgrey'
    color_dict['Emotional'] = 'cyan'
    color_dict['Invitational'] = 'green'
    color_dict['Complete transcripts'] = 'dimgrey'
    
    for typ,_ in text_list:
        
        results = np.array(results_df.loc[results_df.group_by == typ,boxplot_keys])[0].round(2)
        counts = np.array(results_df.loc[results_df.group_by == typ,count_keys])[0]
        total = results_df.loc[results_df.group_by == typ,'num_words'].astype(int).values[0]
        
        z = (1.96,'95% CI')
        z = (2.58,'99% CI')
        z = (3.49,'99.96% CI')
        z = (4.00,'99.996% CI')
        z = (4.50,'99.999% CI')
        # z = (30.0,'>99.9999% CI')
        standard_error = nlpf.calc_standard_error(counts,total,z[0])
        
        if isinstance(standard_errors,pd.DataFrame):
            standard_errors.loc[typ,boxplot_keys] = standard_error
        else:
            standard_errors = pd.DataFrame(np.matrix(standard_error),columns=boxplot_keys)
            standard_errors['type'] = typ
            standard_errors.set_index('type',drop=True,inplace=True)
            
        rects.append(ax.bar(x+shift_list[ctr]*width/len(shift_list),results,width/len(shift_list),color=color_dict[typ],label='{} (n={:,} words)'.format(typ,total)))
        # rects.append(ax.bar(x+shift_list[ctr]*width/len(shift_list),results,width/len(shift_list),yerr=standard_error,capsize=8,label=typ))
        if ctr+1 == len(text_list):
            label='Standard error bars ({}, z={})'.format(z[1],z[0])
        else:
            label = '_none'
        ax.errorbar(x+shift_list[ctr]*width/len(shift_list),results,yerr=standard_error,capsize=8,fmt='D',color='k',label=label)
        ctr+=1
        
    ax.set_ylabel('Percent total words')
    ax.set_xticks(x,boxplot_keys)
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    ax.grid(which='both',axis='y')
    
    # for rect in rects:
    #     ax.bar_label(rect,padding=25,rotation=90)
        
    fig.tight_layout()
    
    plt.show()
    
#%%
    results_df2  = results_df.set_index('group_by',drop=True)
    for typ in count_keys:
        temp_df = pd.DataFrame([list(results_df2[typ]),list(results_df2['num_words']-results_df2[typ])],columns = results_df2.index)
        temp_df['type'] = ['in_corpus','out_of_corpus']
        temp_df.set_index('type',inplace=True,drop=True)
        print('type: {}'.format(typ))
        print('   pvalue={}'.format(stats.chisquare(temp_df).pvalue))

#%%
else:
    print('  pertype = false case')
    
    rcParams.update({'font.size': 22})
    fig,ax = plt.subplots(figsize=[20,15])
    t = pd.melt(results_df,id_vars=['group_by'],value_vars = boxplot_keys,var_name = 'pause type')
    boxplot = sns.boxplot(x='pause type',y='value',data=t,hue='group_by',ax=ax)
    boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel('Percent total words')
    # ax.semilogy()
    
    
    key = boxplot_keys[-2]
    significance = 0.05
    print('\n\nSignificance level: {} (Kruskal-Wallis)'.format(significance))
    if per_patient:
        print('  Aggregating pauses for each patient into a single text block')
    else:
        print('  Treating each pause as its own sample')
    for key in boxplot_keys:
        print('{}'.format(key.upper()))
        if emotional_invitational_separately:
            non_connectional = results_df.loc[results_df.group_by == 'non-connectional',key]
            emotional = results_df.loc[results_df.group_by == 'emotional',key]
            invitational = results_df.loc[results_df.group_by == 'invitational',key]
            transcript = results_df.loc[results_df.group_by == 'Complete transcripts',key]
            no_significance = True
            
            name1 = 'non_connectional'
            name2 = 'emotional'
            no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,non_connectional,emotional,significance)
            name1 = 'non_connectional'
            name2 = 'invitational'
            no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,non_connectional,invitational,significance)
            name1 = 'emotional'
            name2 = 'invitational'
            no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,emotional,invitational,significance)
            
                
            if include_full_transcripts:
                name1 = 'transcript'
                name2 = 'emotional'
                no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,transcript,emotional,significance)
                name1 = 'transcript'
                name2 = 'invitational'
                no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,transcript,invitational,significance)
                name1 = 'transcript'
                name2 = 'non_connectional'
                no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,transcript,non_connectional,significance)
                    
                    
            if no_significance:
                print('  Cannot reject any null hypotheses')
        else:
            non_connectional = results_df.loc[results_df.group_by == 'non-connectional',key]
            connectional = results_df.loc[results_df.group_by == 'connectional',key]
            transcript = results_df.loc[results_df.group_by == 'Complete transcripts',key]
            no_significance = True
            
            name1 = 'non_connectional'
            name2 = 'connectional'
            no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,non_connectional,connectional,significance)
            
                
            if include_full_transcripts:
                name1 = 'transcript'
                name2 = 'connectional'
                no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,transcript,connectional,significance)
                name1 = 'transcript'
                name2 = 'non_connectional'
                no_significance = no_significance & nlpf.compare_distributions_kruskal(name1,name2,transcript,non_connectional,significance)
                    
                    
            if no_significance:
                print('  Cannot reject any null hypotheses')

# stats.kruskal(d1_vals,d2_vals)


    rows = len(text_list)
    cols = len(corpus_list)+1
    
    subfig_size = 5
    fig,ax = plt.subplots(rows,cols,figsize = [cols*subfig_size,rows*subfig_size])
    
    for ctr1,tpl in enumerate(text_list):
        typ,cur_text = tpl
        results = []
        ctr2 = 0
        for ctr,corpus_tpl in enumerate(corpus_list):
            label,corpus = corpus_tpl
            if ctr2 == 0:
                ax[ctr1,ctr2].set_ylabel('{}'.format(typ))
            if corpus == 'temporal_reference':
                for temporal in ['Future','Past']:
                    
                    # fig2,ax2 = plt.subplots()
                    temp = results_df.loc[results_df['group_by'] == typ,:]
                    data = temp[temporal]
                    ax[ctr1,ctr2].hist(data,bins=20)
                    if ctr1 == 0:
                        ax[ctr1,ctr2].set_title('{}'.format(temporal)) 
                    ctr2 +=1
                    
            else:
                temp = results_df.loc[results_df['group_by'] == typ,:]
                data = temp[label]
                
                ax[ctr1,ctr2].hist(data,bins=20)
                if ctr1 == 0:
                    ax[ctr1,ctr2].set_title('{}'.format(label)) 
                ctr2 +=1 



results_fn = 'word_count_table'

if per_type:
    results_fn = '{}_per-type'.format(results_fn)
else:
    if per_patient:
        results_fn = '{}_per-patient'.format(results_fn)
    else:
        results_fn = '{}_per-pause'.format(results_fn)
    
if emotional_invitational_separately:
    results_fn = '{}_inv-emo-sep'.format(results_fn)
else:
    results_fn = '{}_inv-emo-combined'.format(results_fn)


results_fn = '{}.csv'.format(results_fn)

results_df.to_csv(os.path.join(cd,results_fn))













