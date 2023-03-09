#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:47:57 2019

@author: lclarfel
"""

import numpy as np
import pandas as pd
from Conversation import Primary_PCCRI_Conversation
from matplotlib import pyplot as plt
import nltk
nltk.download('averaged_perceptron_tagger')
from Patients import Patients

# Basically just a list of conversations, but allows basic common operations on
# that list without requiring lots of loops or list concatenations   
class Conversations():
    def __init__(self, convs):
        self.convs = convs
        self.features = {}
        
    def addConv(self, conv):
        """Add a new conversation
    
        Parameters:
        conv: The Conversation instance to be added
       """
        if conv not in self.convs:
            self.convs.append(conv)

    def removeConv(self, conv):
        """Remove a conversation
    
        Parameters:
        conv: The Conversation instance to be removed
       """
        if conv in self.convs:
            self.convs.remove(conv)
    
    def getHUscores(self, scoreType):
        """Get the heard/understood "score" for all conversations that have them
    
        Parameters:
        scoreType: How to calculate the heard/understood scores. Options include:
            before - The "before" score
            after - The "after" score
            delta - The difference between before and after (negative is good)
            imp_or_max - True if the score was perfect before or improved
            maggie - Computes the "maggie metric", as defined elsewhere
       """

        hu_scores = []
        for conv in self.convs:
            if type(conv) == Primary_PCCRI_Conversation:
                hu_scores.append(conv.getHUscore(scoreType))
            else:
                hu_scores.append(np.nan)
        return hu_scores
    
    def getPIDs(self):
        """Get all patient ID's
        
        Returns:
        Ordered list of all patient ID's
       """
        pids = []
        for conv in self.convs:
            pids.append(conv.pid)
        return pids
    
    def getCIDs(self):
        """Get all conversation numbers
        
        Returns:
        Ordered list of all conversation numbers
       """
        cids = []
        for conv in self.convs:
            cids.append(conv.cid)
        return cids
    
    # FIX: Currently set up only for PCCRI, need to edit to accept any key
    def getConv(self, pid, cid):
        """Get all patient ID's
       """
        for conv in self.convs:
            if conv.pid==pid and conv.cid==cid:
                return conv
        print('Patient '+str(pid)+', Conv. '+str(cid)+' not found')
        return []

    # FIX: Currently set up only for PCCRI, need to edit to accept any key
    def getConv_i(self, pid, cid):
        """Returns index number of a specified conversation
    
        Parameters:
        pid: Patient ID
        cid: Conversation Number
    
        Returns:
        index of the specified conversation
       """
        for i in range(len(self.convs)):
            conv = self.convs[i]
            if conv.pid==pid and conv.cid==cid:
                return i
        print('Patient '+str(pid)+', Conv. '+str(cid)+' not found')
        return []
    
    def get_primary_convs(self):
        """Returns the subset of all conversations that are "Primary Conversations"
    
        Returns:
        A conversations object containing only "Primary Conversations"
       """
        # return a new Conversations instance with just primary conversations
        primary_convs = Conversations([conv for conv in self.convs if type(conv)==Primary_PCCRI_Conversation])
        # NOTE: Some features in primary_convs.features may not appear in any
        # primary conversation. Maybe add an update to exclude such features?
        primary_convs.features = self.features
        return primary_convs
    
    def getNumTurns(self):
        """Returns the number turns in each conversation
    
        Returns:
        The number of turns in each conversation
       """
        nTurns = []
        for conv in self.convs:
            nTurns.append(len(conv.turns))
        return nTurns
    
    def add_patient_features(self, patients):
        conv_indices = []
        for patient in patients.p_list:
            pid_convs = np.where(np.array(self.getPIDs()) == patient.pid)[0]
            conv_indices.append(pid_convs)
        
        # Remove the "patient ID" header so it isn't added as a feature
        patients.headers = np.array([col for col in patients.headers if col != 'p0_patient_id'])            
        
        for feature in patients.headers:
            for patient_i, indices in zip(range(len(conv_indices)), conv_indices):
                for i in indices:
                    self.convs[i].add_feature(feature, patients.p_list[patient_i].metadata[feature])
    
    def add_features(self, fName, pids, cids, feature):
        """Add a set of features to some (or all) conversations. The inputs
        `pids`, `cids`, and `feature` should all be parallel lists such that
    
        Parameters:
        fName: The name of the new feature to be added
        pids: Patient ID's
        cids: Conversation Numbers
        feature: The new feature to be added
       """
        for conv in self.convs:
            i = [i for i,pid,cid in zip(range(len(self.convs)),self.getPIDs(),self.getCIDs()) if pid == conv.pid and cid == conv.cid]
            if len(i) == 1:
                conv.add_feature(fName,feature[i[0]])
            elif len(i) > 1:
                print('weird... more than 1 match')
            else:
                conv.addFeature(fName,None)
                print("Feature '"+fName+"' not found for PID: "+str(conv.pid)+'; Conv: '+str(conv.cid))
    
    def getFeature(self, fNames):
        """Get a specified feature (or set of features) for all conversations
    
        Parameters:
        fName: Name of feature (str) or features (list) to be retrieved
        
        Returns:
        The specified feature(s)
       """
        if type(fNames) == str or len(fNames)==1:
            feature = []
            for conv in self.convs:
                if fNames in conv.features.keys():
                    feature.append(conv.features[fNames])

        elif len(fNames)>1:
            feature = [[] for _ in range(len(fNames))]
            for i in range(len(fNames)):
                for conv in self.convs:
                    if fNames[i] in conv.features.keys():
                        feature[i].append(conv.features[fNames[i]])
            
        return feature
    
    def import_feature_descrip(self, file_name, *ignore_features):
        """Import a set of feature descriptions from an external file. Both
        csv and excel formats are accepted, but must be properly formatted with
        three columns:
            "feature_name" - Name of the feature
            "description" - Written description of the feature
            "data_type" - Must include words numerical,categorical,discrete,continuous
    
        Parameters:
        file_name: The path and filename containing the feature descriptions
        ignore_features: (optional) list of columns to ignore
       """
        # NOTE: the encoding specified hopefully removes 'special characters' 
        #       such as \ufeff (https://github.com/clld/clldutils/issues/65)
        #       This fix has not been fully tested and deserves another look.
        if file_name[-3:] == 'csv':
            data = pd.read_csv(file_name) #,encoding='utf-8-sig')
        elif file_name[-3:] == 'xls' or file_name[-4:] == 'xlsx':
            data = pd.read_excel(file_name) #,encoding='utf-8-sig')   
        
        for feature in list(data['feature_name']):
            if feature not in ignore_features:
                self.features[feature] = (data.loc[data.feature_name == feature]['description'].values[0],
                                          data.loc[data.feature_name == feature]['data_type'].values[0])
    
    # ADD: Currently set up only for PCCRI, need to add fct to accept any key
    def import_features_PCCRI(self, file_name, p_key, c_key, *ignore_cols):
        """Import a set of features from an external file. Both csv and excel 
        formats are accepted. Each row represents a converastion and columns 
        must include patient ID, conversation number, and any new features to
        be added. 
    
        Parameters:
        file_name: The path and filename containing the features
        p_key: Column header for the spreadsheed column with Patient ID's
        c_key: Column header for the spreadsheed column with Conversation #'s
        ignore_features: (optional) list of columns to ignore
       """
        # Add features to a PCCRI conversation object from a spreadsheet. 
        #
        # INPUTS:
        #  - file_name: the name of the spreadsheef file, including extension.
        #               acceptable inputs: csv, xls, or xlsx
        #  - p_key: Column header of the patient ID
        #  - c_key: Column header of the conversation number
        #  - ignore_cols: (optional) headers of columns not to include  (list)
        
        if file_name[-3:] == 'csv':
            data = pd.read_csv(file_name)
        elif file_name[-3:] == 'xls' or file_name[-4:] == 'xlsx':
            data = pd.read_excel(file_name)

        cols = list(data.columns)
        if p_key in cols:
            cols.remove(p_key)
        else:
            print(p_key, 'not found as column header')
            
        if c_key in cols:
            cols.remove(c_key)
        else:
            print(c_key, 'not found as column header')
            
        if ignore_cols:
            for col in ignore_cols[0]:
                if col in cols:
                    cols.remove(col)
                else:
                    print(col, 'not found as column header')

        for index, row in data.iterrows():
            i = self.getConv_i(row[p_key],row[c_key])
            if i:
                for col in cols:
                    if isinstance(row[col],str) or not np.any(np.isnan(row[col])):
                        self.convs[i].features[col] = row[col]
            else:
                print('Features not added')
    
    def export_features(self, file_name):
        """Saves all features in a spreadsheet for external analysis. Current
        accepted formats are csv and excel, specified by the extention used in
        the file name.
    
        Parameters:
        file_name: The path and filename of the spreadsheet to be produced
       """
        df = pd.DataFrame([{**conv.key, **conv.features} for conv in self.convs])
        if file_name[-3:] == 'csv':
            df.to_csv(file_name)
        elif file_name[-3:] == 'xls' or file_name[-4:] == 'xlsx':
            # This may not be working yet
            print('WARNING: This may not be working yet, and may require extra packages')
            df.to_excel(file_name)
        else:
            print('ERROR: File extension not recovnized')
        
    def list_features(self):
        """Print a list of all features, with sample sizes (the number of 
        conversations for which each feature is specified)
        
        Returns:
        An alphabetized list of all conversational features (with sample sizes)
       """
        all_features = {}
        for conv in self.convs:
            for fName in conv.features.keys():
                if conv.features[fName] != np.nan:
                    if fName in all_features.keys():
                        all_features[fName].append(conv.features[fName])
                    else:
                        all_features[fName] = [conv.features[fName]]
        
        
        feature_list = list(all_features)
        feature_list.sort()
        for feature_name in feature_list:
            print(feature_name + ' (N=' + str(len(all_features[feature_name])) + ')')
        return feature_list
        
    def describe_feature(self, feature_name):
        """For a specified feature, print a description of the feature and
        display standardized statistics / histograms to summarize the data
    
        Parameters:
        feature_name: The name of the feature to be summarized/described
       """
        feature_data = []
        for conv in self.convs:
            if feature_name in conv.features.keys():
                feature_data.append(conv.features[feature_name])
        
        if len(feature_data) == 0:
            print('Feature ' + feature_name + ' not found in any conversation')
        elif feature_name not in self.features.keys():
            print('No description available for feature ' + feature_name + ' (N=' + str(len(feature_data)) + ')')
        elif 'categorical' in self.features[feature_name][1]:
            print('Feature Name: ' + feature_name)
            print('Feature Description: ' + self.features[feature_name][0])
            print('Feature Data Type: ' + self.features[feature_name][1])
            plt.figure()
            plt.hist(feature_data)
            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
        elif 'numerical' in self.features[feature_name][1]:
            print('Feature Name: ' + feature_name)
            print('Feature Description: ' + self.features[feature_name][0])
            print('Feature Data Type: ' + self.features[feature_name][1])
            print('Mean: ' + str(np.mean(feature_data)))
            print('Median: ' + str(np.median(feature_data)))
            print('Standard Deviation: ' + str(np.std(feature_data)))
            print('Minimum: ' + str(np.min(feature_data)))
            print('Maximum: ' + str(np.max(feature_data)))
            # For discrete numerical data w/ less than 10 values, each value == a bin
            vals = np.unique(feature_data)
            if 'discrete' in self.features[feature_name][1] and len(vals) < 10:
                plt.figure()
                plt.bar(vals,[len([1 for f in feature_data if f == vals[i]]) for i in range(len(vals))])
                plt.xlabel(feature_name)
                plt.ylabel('Frequency')
            else:
                plt.figure()
                plt.hist(feature_data)
                plt.xlabel(feature_name)
                plt.ylabel('Frequency')
                
    def plot_Lindsay_curve(self, word_list, nbins, title, show_plot):
        """Given a word list, plot (and return) a frequency histogram of word 
        occurrance over narrative time (aka, "Lindsay Curves"). 
    
        Parameters:
        word_list: word (str) or list of words (list)
        nbins: Number of x-iles (bins in the histogram)
        title: Title of the figure
        show_plot: Whether to display the histogram
        
        Returns:
        Tuple of bin counts and bin boundaries for the created histogram
       """

        # If input is a single word, make sure it is in a list   
        if isinstance(word_list,str):
            word_list = [word_list]

        word_at = []
        for conv in self.convs:    
            all_words = np.hstack(conv.turns)
            for word in word_list:
                i = np.where(all_words==word)[0]
                if len(i):
                    word_at.append(i/all_words.shape[0])
        
        word_at = np.hstack(word_at)
        plt.figure()
        p = plt.hist(word_at,bins=nbins)
        if not show_plot:
            plt.clf()
        else:
            plt.xlabel(str(nbins) + '-iles')
            plt.ylabel('Frequency')
            plt.title(title + ' (N='+str(len(word_at))+')')
        return p
            
    
    def add_temporal_features(self, nbins):
        future_ALL = []
        past_ALL = []
        present_ALL = []
        
        for conv in self.convs:
            future_count = []
            present_count = []
            past_count = []
            
            # Temporal Reference
            tags = nltk.pos_tag(list(np.hstack(conv.turns)))
            for i in range(len(tags)): 
                try: 
                    if tags[i][1] == 'VB':
                        # future, infinitive, future imperative
                        if tags[i-1][1] == 'MD' or tags[i-1][0] == 'to' or tags[i-1][0] == "let's" or (tags[i-2][0] == "let" and tags[i-1][0] == "us"):
                            future_count.append(i)
                        # present simple
                        # else: 
                        #     present_count += 1 
                        #     tot_present_count += 1
                        
                    elif tags[i][1] == 'VBD':
                        # past simple/prereterite
                        past_count.append(i)
                        
                    elif tags[i][1] == 'VBG':
                        # future perfect continuous
                        if tags[i-3][0] == 'will' and tags[i-2][0] == 'have' and tags[i-1][0] == 'been':
                            future_count.append(i)
                        # present continuous
                        elif tags[i-1][0] == "am" or tags[i-1][0] == "are" or tags[i-1][0] == "is":
                            present_count.append(i)
                        # past continuous
                        elif tags[i-1][0] == "was" or tags[i-1][0] == "were":
                            past_count.append(i)
                        # future continuous
                        elif tags[i-2][1] == 'MD' and tags[i-1][0] == 'be':
                            future_count.append(i)
                        # present perfect continuous 
                        elif (tags[i-2][0] == "has" or tags[i-2][0] == "have") and tags[i-1][0] == "been":
                            present_count.append(i)
                        # past perfect continuous
                        elif tags[i-2][0] == "had" and tags[i-1][0] == "been":
                            past_count.append(i)
                        # present participle
                        else:
                            present_count.append(i)
                        
                    elif tags[i][1] == 'VBN':
                        # future perfect 
                        if tags[i-2][0] == "will" and tags[i-1][0] == "have":
                            future_count.append(i)
                        # present perfect, past perfect, perfect participle
                        else: 
                            past_count.append(i)
                    # present simple
                    elif tags[i][1] == 'VBP':
                        present_count.append(i)
                        
                
                    elif tags[i][1] == 'VBZ':
                        present_count.append(i)
                except IndexError:
                    print(tags[i][0], "not categorized, truncated information")
                    
            past_ALL.append(np.array(past_count)/sum([len(turn) for turn in conv.turns]))
            future_ALL.append(np.array(future_count)/sum([len(turn) for turn in conv.turns]))
            present_ALL.append(np.array(present_count)/sum([len(turn) for turn in conv.turns]))
            
            edges = np.arange(nbins+1)/nbins
        
            bincounts_pa = [np.sum(np.logical_and(past_ALL[-1] > edges[i], past_ALL[-1] < edges[i+1])) for i in range(nbins)]
            bincounts_pr = [np.sum(np.logical_and(present_ALL[-1] > edges[i], present_ALL[-1] < edges[i+1])) for i in range(nbins)]
            bincounts_fu = [np.sum(np.logical_and(future_ALL[-1] > edges[i], future_ALL[-1] < edges[i+1])) for i in range(nbins)]

            for i in range(nbins):
                # QUESTION: Should we normalize, as done here, or give raw counts?
                conv.features['past_'+str(i+1)+'_of_'+str(nbins)] = bincounts_pa[i]/sum(bincounts_pa)
                conv.features['present_'+str(i+1)+'_of_'+str(nbins)] = bincounts_pr[i]/sum(bincounts_pr)
                conv.features['future_'+str(i+1)+'_of_'+str(nbins)] = bincounts_fu[i]/sum(bincounts_fu)
                    
        for i in range(nbins):
            # QUESTION: Should we normalize, as done here, or give raw counts?
            self.features['past_'+str(i+1)+'_of_'+str(nbins)] = (
                    'Percentage of past tense between percentiles ' + str(edges[i]*100) + ' and ' + str(edges[i+1]*100),
                    'numerical,continuous')
            self.features['present_'+str(i+1)+'_of_'+str(nbins)] = (
                    'Percentage of present tense between percentiles ' + str(edges[i]*100) + ' and ' + str(edges[i+1]*100),
                    'numerical,continuous')
            self.features['future_'+str(i+1)+'_of_'+str(nbins)] = (
                    'Percentage of future tense between percentiles ' + str(edges[i]*100) + ' and ' + str(edges[i+1]*100),
                    'numerical,continuous')

        plt.figure()
        plt.hist(np.hstack(past_ALL),bins=nbins)
        plt.title('Past Tense')
        plt.figure()
        plt.hist(np.hstack(future_ALL),bins=nbins)
        plt.title('Future Tense')
        plt.figure()
        plt.hist(np.hstack(present_ALL),bins=nbins)
        plt.title('Present Tense')
    # ADD METHOD TO REMOVE ANY CONVERSATION WITH FEWER THAN xx SPEAKER TURNS
