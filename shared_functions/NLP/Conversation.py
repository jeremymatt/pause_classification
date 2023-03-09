#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:24:43 2019

@author: lclarfel
"""

import nlpFunctions as nlp
import os
#import drawConversation as draw

class Conversation:
    def __init__(self, key, turns, speakers, speaker_labels):
        # ALWAYS use dictionary for compatibility (allows composite keys)
        if type(key) == dict:
            self.key = key
        else:
            self.key = {'Conversation_ID':key}
        self.turns = turns
        self.speakers = speakers
        self.speaker_labels = speaker_labels
        self.features = {}
        
    def __str__(self):
        if type(self.key) == dict:
            return ', '.join("{}: {}".format(k, v) for k, v in self.key.items())
        else:
            return 'Conversation ID: ' + str(self.key)
    
    def add_feature(self, fName, fData):
        self.features[fName] = fData
        
    def getTurnLengths(self):
        return [len(turn) for turn in self.turns]
    
    def replaceWords(self, oldWords, newWord):
        self.turns = nlp.replaceWords(self.turns, oldWords, newWord)
        return self
    

class PCCRI_Conversation(Conversation):
    
    def __init__(self, folder, filename, convOpt):
        
        # Default options:
        #  - Merge consecutive speaker turns (True)
        #  - Only include primary conversations (False)
        #  - Remove any words from the supplied list (None)
# =============================================================================
#         print(argv)
#         if ~len(argv):
#             convOpt = {'merge':True, 'primary':False, 'stopWords':None}
#         else:
#             convOpt = argv
# =============================================================================
        
        data = nlp.getLines(os.path.join(folder,filename))
        data = data[5:-1] # remove header/footer
        
        # Labels: Patient==0, Clinician==1, Other==2
        data, labels = nlp.prepData(data,convOpt)
                
        if convOpt['stop_words']:
            data = nlp.removeWords(data,convOpt['stop_words'])
        
        #print(convOpt['merge'])
        if convOpt['merge']:
            data,labels = nlp.mergeConsectutiveTurns(data,labels)
            
        conv_ID = filename[5]
        breakhere=1
        if (ord(conv_ID)>=48) and (ord(conv_ID)<=57):
            cid = int(conv_ID)
        else:
            cid = ord(conv_ID)-ord('A')+1 # Conversation Number
        
        self.pid = int(filename[:4])
        self.cid = cid
        
        self.key = {'Patient':self.pid, 'Conversation':self.cid}
        
        self.turns = data
        self.speakers = labels
        self.speakerLabels = {'patient':0, 'clinician':1,'other':2}
        self.features = {}
    
class Primary_PCCRI_Conversation(PCCRI_Conversation):
    
    def __init__(self, folder, filename, convOpt, hub, hua):
        super().__init__(folder, filename, convOpt)
        self.hub = hub
        self.hua = hua
    
    # Get heard/understood score
    def getHUscore(self, score_type):
        # Change in score (int)
        if score_type == 'before':
            return self.hub
        elif score_type == 'after':
            return self.hua
        elif score_type == 'delta':
            return self.hua - self.hub
        # Whether the score improved, or was at max (boolean)
        elif score_type == 'imp_or_max':
            return (self.hua < self.hub) or (self.hua == 1)
        elif score_type == 'maggie':
            change = self.hua - self.hub
            bestPossibleChange = 1-self.hub
            worstPossibleChange = 5-self.hub
            rawfit = (change - (worstPossibleChange-bestPossibleChange)) / (
                    worstPossibleChange - bestPossibleChange)
            return rawfit/2 + self.hua
        else:
            print('ERROR: ' + score_type + 'is an unknown heard/understood score')
            return []
