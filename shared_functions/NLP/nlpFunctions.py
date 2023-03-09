# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:54:45 2019

A collection of customized NLP functions for preparing and analyzing transcripts 

@author: lclarfel; adapted & extended by jmatt
"""

import re
import itertools
import numpy as np
from scipy import spatial
import os
import nltk
from scipy import stats
nltk.download('averaged_perceptron_tagger')

# Open a transcript and read in the text line-by-line
def getLines(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            #if line has something in it
            if line != '\n':
                data.append(line)
    return data

def removeComments(data):
    for i in range(len(data)):
        turn = data[i]
        # Take care of square brackets:
        while '[' in turn:
            startPos = turn.find('[')
            endPos = turn.find(']')
            turn = ''.join([turn[i] for i in range(len(turn)) if i < startPos or i > endPos])
        while '{' in turn:
            startPos = turn.find('{')
            endPos = turn.find('}')
            turn = ''.join([turn[i] for i in range(len(turn)) if i < startPos or i > endPos])
        #while '<' in turn:
        #    startPos = turn.find('<')
        #    endPos = turn.find('>')
        #    turn = ''.join([turn[i] for i in range(len(turn)) if i < startPos or i > endPos])
        data[i] = turn
    return data

# split lines by word, clean punctuation/capitalization, and label by speaker
# NOTE: header should be removed
def prepData(data,convOpt):
    if data[0][:9] == 'PatientID':
        raise Exception('ERROR: The header has not been removed')
        
    # Remove any "comments" ([],{},<>)

    if convOpt['remove_nonAlphabets']:
        data = removeComments(data)
    
    line_labels = [0 for i in range(len(data))] # initialize labels
    
    # regex = re.compile('[^a-zA-Z]') # This approach seems to miss the mark
    
    # For each line...
    for i in range(len(data)):
        # Figure out who's speaking
        whosTurn = data[i][0]
        if whosTurn == 'P':
            line_labels[i] = 0
        elif whosTurn == 'C':
            line_labels[i] = 1
        else:
            line_labels[i] = 2
            
        # Convert to lowercase
        data[i] = data[i].lower()

        if convOpt['remove_nonAlphabets']:
        # Remove stuff in brackets of all types '( )[ ]< >{ }'
            data[i] = re.sub('\(.*?\)',"",data[i])
            data[i] = re.sub('\[.*?\]',"",data[i])
            data[i] = re.sub('\{.*?\}',"",data[i])
            #data[i] = re.sub('\<.*?\>',"",data[i])

            # Replace dashes (-), periods (.), apostrophes (') and slashes (/) with spaces
            data[i] = re.sub("[-./']",' ',data[i])
            #data[i] = re.sub('[.]',' ',data[i])
            #data[i] = re.sub('[/]',' ',data[i])
            #data[i] = re.sub("[']",' ',data[i])

            # Remove other punctuation (,?!";:)
            data[i] = re.sub('[,?!";:]','',data[i])
        
        # replace double spaces with single ones
        data[i] = re.sub('[ ]{2,}',' ',data[i])
# =============================================================================
#         while '  ' in data[i]:
#             data[i] = re.sub('  ',' ',data[i])
# =============================================================================
        
                
        # Split the turn into words
        words = str.split(data[i])
        
        # Drop the first word (speaker ID)
        data[i] = words[1:]
        
        # Join the words back together
        # data[i] = ' '.join(words)
        # data[i] = [regex.sub('',word).lower() for word in words[1:]] #Not sure this is the best approach
        
        
    
    return data, line_labels

# Returns the word count broken down by speaker (clinician/patient/other)
def getWordCounts(data,labels,wordList):
    
    wordCount = [0,0,0]
    
    for (line,label) in zip(data,labels):
        for word in line:
            if word in wordList:
                wordCount[label] += 1
                
    return wordCount

# Get the word counts from a word list by decile
def getWordCountsByDecile(data,wordList,numDec):
    cts = [0 for _ in range(numDec)]
    # If we have a "list of lists", merge them
    if len(data)>0 and type(data[0]) == list:
        data = list(itertools.chain.from_iterable(data))
    numwords = len(data)
    numperdec = numwords//numDec
    decCts = np.tile(numperdec,numDec)
    r = np.random.permutation(numDec)
    for i in range(numwords - numperdec*numDec):
        decCts[r[i]] += 1
    
    decCts = np.cumsum(decCts)
    decCts = np.hstack(([0],decCts))   
    
    for i in range(numDec):
        subwordlist = data[decCts[i]:decCts[i+1]]
        
        # This block will be where I add code to ignore "Thank You's" when counting "you's"
        for word in wordList:
            cts[i] += np.sum([1 for w in subwordlist if w == word])
    return cts
    
# Split the text of a conversation by patient vs. clinician    
def splitByPandC(data, labels):
    cWords = []
    pWords = []
    oWords = []
    for (words,label) in zip(data,labels):
        if label == 0:
            pWords += words
        elif label == 1:
            cWords += words
        else:
            oWords += words
    return pWords, cWords, oWords

# If a patient/clinician has consecutive speaker turns, merge them
def mergeConsectutiveTurns(data,labels):
    data_new = [data[0]]
    labels_new = [labels[0]]
    for i in range(len(data)-1):
        if labels[i] == labels[i+1]:
            data_new[-1] += data[i+1]
        else:
            data_new.append(data[i+1])
            labels_new.append(labels[i+1])
    return data_new, labels_new

#add dynamic turn taking series here
#to attempt to capture patient clinician turns only


# Get a count for every word, add it to a growing dictionary of counts
def getWordCountsALL(data, wordDict):
    if len(data)>0 and type(data[0]) == list:
        data = list(itertools.chain.from_iterable(data))
    for word in data:
        if word in list(wordDict.keys()):
            wordDict[word] += 1
        else:
            wordDict[word] = 1
    return wordDict

# Get lead counts for words in the wordlist
def getLeadCounts(data,labels):
    pturns = 0
    cturns = 0
    turnNum = 0
    
    totPwords = 0
    totCwords = 0
            
    pDict = {}
    cDict = {}
    
    pLeadWords = {}
    cLeadWords = {}
    
    for i in range(len(data)):
        whosTurn = labels[i]
    
        if whosTurn == 0:
            pturns += 1
        elif whosTurn == 1:
            cturns += 1
            

        words = list(set(data[i]))
        for j in range(len(words)):
            word = words[j]
            if word not in pDict.keys():
                pDict[word] = []
                cDict[word] = []
                
            if whosTurn == 0:
                pDict[word].append(turnNum)
                totPwords += 1
            elif whosTurn == 1:
                totCwords += 1
                cDict[word].append(turnNum)
            #else:
                #print(whosTurn)
                
        turnNum += 1
    
    pLeadCt = [0 for i in range(turnNum)]
    cLeadCt = [0 for i in range(turnNum)]
        
    for (pKey, pVal), (cKey, cVal) in zip(pDict.items(),cDict.items()):
        
        if pKey != cKey:
            print('error')
        
        if pVal and cVal:
            for pSpot in pVal:
                gap = [pSpot-cSpot for cSpot in cVal]
                if -1 in gap:
                    pLeadCt[pSpot] += 1
                    if pKey in list(pLeadWords.keys()):
                        pLeadWords[pKey] += 1
                    else:
                        pLeadWords[pKey] = 1
                    #cFollowWords[pSpot+1] += 1
                
                if 1 in gap:
                    #pFollowWords[pSpot-1] += 1
                    cLeadCt[pSpot] += 1
                    if cKey in list(cLeadWords.keys()):
                        cLeadWords[cKey] += 1
                    else:
                        cLeadWords[cKey] = 1
    
    return pLeadWords, cLeadWords, pLeadCt, cLeadCt
     
def getAccumCurve(data):
    allWords = np.hstack(data)
    wordDict = {}
    ct = 1
    curve = np.zeros(len(allWords))
    curve2 = []
    
    for i,word in zip(range(len(allWords)),allWords):
        if word in wordDict:
            curve[i] = wordDict[word]
        else:
            curve[i] = ct
            wordDict[word] = ct
            ct += 1
            curve2.append(i)
    
    return curve, curve2
    
def leadByWords(data,labels,numDec):
    
    isLeading = [0 for _ in range(len(data))]
    for i in range(1,len(data)):
        if len(data[i-1]) > len(data[i]):
            isLeading[i-1] = 1
            
    numwords = len(data)        
    numperdec = numwords//numDec
    decCts = np.tile(numperdec,numDec)
    r = np.random.permutation(numDec)
    for i in range(numwords - numperdec*numDec):
        decCts[r[i]] += 1
    
    decCts = np.cumsum(decCts)
    decCts = np.hstack(([0],decCts))   
    
    pLeading = [isLeading[i] for i in range(len(isLeading)) if labels[i] == 0]
    cLeading = [isLeading[i] for i in range(len(isLeading)) if labels[i] == 1]
    
    whosLeading = [0 for _ in range(numDec)]
    
    for i in range(numDec):
        isLead_sub = isLeading[decCts[i]:decCts[i+1]]
        lbl_sub = labels[decCts[i]:decCts[i+1]]
        
        p_sub = [isLead_sub[i] for i in range(len(isLead_sub)) if lbl_sub[i] == 0]
        c_sub = [isLead_sub[i] for i in range(len(isLead_sub)) if lbl_sub[i] == 1]
        
        if sum(c_sub) > sum(p_sub):
            whosLeading[i] = 1
    
    return whosLeading

def replaceWords(data, wordList, replaceWord):
    for i in range(len(data)):
        data[i] = [word if word not in wordList else replaceWord for word in data[i]]
    return data

def removeWords(data, wordList):
    for i in range(len(data)):
        data[i] = [word for word in data[i] if word not in wordList]
    return data
        


    
    
def getEdgeWeight(line0, line1, vecDict):
    edgeweight = 0
    for word0 in line0:
        for word1 in line1:
            if word0 in vecDict and word1 in vecDict:
                w = spatial.distance.cosine(vecDict[word0],vecDict[word1])
                # print('weight from ', word0, ' to ', word1, ' is ', w)
                if w == 0:
                    edgeweight += 10
                elif w < 0.7:
                    edgeweight += 1/w
    return edgeweight


def get_patient_sample_counts(df,pid,patient_col,label_col):
    temp = df.loc[df[patient_col] == pid,:]
    
    types = set(temp[label_col])
    
    count_dict = {}
    
    for typ in types:
        count_dict[typ] = sum(temp[label_col]==typ)
        
    return count_dict

def compare_actual_expected_samples(expected,actual,transcripts_path):
    output_path = os.path.split(transcripts_path)[0]
    
    actual_patient_col = 'Patient'
    actual_label_col = 'label True'
    expected_patient_col = 'PID'
    expected_label_col = 'silencetype_actual'
    
    actual_pids = set(actual[actual_patient_col])
    expected_pids = set(expected[expected_patient_col])
    
    extra_actual_pids = actual_pids.difference(expected_pids)
    extra_expected_pids = expected_pids.difference(actual_pids)
    marked_expected_pids = actual_pids.intersection(expected_pids)
    
    actual_pids = sorted(list(actual_pids))
    expected_pids = sorted(list(expected_pids))
    extra_actual_pids = sorted(list(extra_actual_pids))
    extra_expected_pids = sorted(list(extra_expected_pids))
    marked_expected_pids = sorted(list(marked_expected_pids))
    
    
    with open(os.path.join(output_path,'expected_actual_comparison.txt'),'w') as f:
        matches = True
        
        f.write('Conversations with marked pauses in transcripts that should not have pauses:\n')
        for pid in extra_actual_pids:
            matches = False
            f.write('  PID: {}\n'.format(pid))
            count_dict = get_patient_sample_counts(actual,pid,actual_patient_col,actual_label_col)
            for key in sorted(list(count_dict.keys())):
                f.write('    Type {}: {}\n'.format(int(key),count_dict[key]))
                
        f.write('\nConversations that should have had pauses marked but were not found in transcripts:\n')
        for pid in extra_expected_pids:
            matches = False
            f.write('  PID: {}\n'.format(pid))
            count_dict = get_patient_sample_counts(expected,pid,expected_patient_col,expected_label_col)
            for key in sorted(list(count_dict.keys())):
                f.write('    Type {}: {}\n'.format(int(key),count_dict[key]))
                
        f.write('\nConversations that with pause in transcripts that should have pauses in transcripts:\n')
        for pid in marked_expected_pids:
            actual_count_dict = get_patient_sample_counts(actual,pid,actual_patient_col,actual_label_col)
            expected_count_dict = get_patient_sample_counts(expected,pid,expected_patient_col,expected_label_col)
            actual_types = set(actual_count_dict.keys())
            expected_types = set(expected_count_dict.keys())
            
            actual_diff = actual_types.difference(expected_types)
            expected_diff = expected_types.difference(actual_types)
            
            for typ in actual_diff:
                expected_count_dict[typ] = 0
                
            for typ in expected_diff:
                actual_count_dict[typ] = 0
                
            cur_matches = write_counts(expected_count_dict,actual_count_dict,pid,f)
            
            matches = matches & cur_matches
            
        if matches:
            print('\nExpected number & type of samples found for all patients\n')
        else:
            print('\nAt least one sample has a mismatch in expected and actual sample counts/types')
            print('    Check the output log at:\n  {}'.format(os.path.join(output_path,'expected_actual_comparison.txt')))
            
            
            
def write_counts(expected_count_dict,actual_count_dict,pid,f):
    matches = True
    f.write('\n  PID: {}\n'.format(int(pid)))
    p0 = 'Type '
    p1 = ' Expected '
    p2 = ' Actual '
    p3 = ' Error'
    l0 = len(p0)
    l1 = len(p1)
    l2 = len(p2)
    l3 = len(p3)
    sep = '|'
    spacer = '-'
    write_line(p0,p1,p2,p3,l0,l1,l2,l3,sep,spacer,f)
    
    p0 = ''
    p1 = ''
    p2 = ''
    p3 = ''
    sep = '+'
    spacer = '-'
    write_line(p0,p1,p2,p3,l0,l1,l2,l3,sep,spacer,f)
    
    sep = '|'
    spacer = ' '
    
    for key in sorted(list(expected_count_dict.keys())):
        p0 = '{}'.format(int(key))
        p1 ='{}'.format(int(expected_count_dict[key]))
        p2 = '{}'.format(int(actual_count_dict[key]))
        if expected_count_dict[key] == actual_count_dict[key]:
            p3 = ''
        else:
            p3 = ' ERROR'
            matches = False
            
        write_line(p0,p1,p2,p3,l0,l1,l2,l3,sep,spacer,f)
    
    return matches
    
    
def write_line(p0,p1,p2,p3,l0,l1,l2,l3,sep,spacer,f):
    f.write('    {}{}{}{}{}{}{}\n'.format(p0.center(l0,spacer),sep,p1.center(l1,spacer),sep,p2.center(l2,spacer),sep,p3.ljust(l3,spacer)))
            

def remove_non_letters(text):
    if isinstance(text,list):
        text = ' '.join(text)
        
    if not isinstance(text,str):
        print('ERROR: text must be either list of strings or a string')
        return None
    
    #Match any character that is not a lower case letter, an upper case letter, or a space
    pat = r'[^a-zA-Z ]'
    text = re.sub(pat,'',text)
    
    return text
    
            
def corpus_word_counts_by_bin(word_corpus,input_transcript,n_bins):
    if type(input_transcript) == str:
        input_transcript = input_transcript.split(' ')
    
    num_words = len(input_transcript)
    
    bin_divisions = list(np.round(np.linspace(0,num_words,n_bins+1),0).astype(int))
    
    index_tuples = list(zip(bin_divisions[:-1],bin_divisions[1:]))
    
    bin_count_dict = {}
    
    for ctr,tpl in enumerate(index_tuples):
        start,end = tpl
        text = remove_non_letters(input_transcript[start:end])
        text = text.lower()
        # print('current text: {}'.format(text))
        if word_corpus == 'temporal_reference':
            future_count,present_count,past_count = count_past_present_future(text)
            bin_count_dict[ctr] = {}
            bin_count_dict[ctr]['past'] = past_count
            bin_count_dict[ctr]['present'] = present_count
            bin_count_dict[ctr]['future'] = future_count
        else:
            bin_count_dict[ctr] = count_corpus_hits_in_string(word_corpus, text)
            
            # print('Found {} total matches for bin {}'.format(bin_count_dict[ctr]['total_count'],ctr))
      
    bin_count_dict['total_transcript_words'] = num_words
    
    return bin_count_dict
        

def count_corpus_hits_in_string(word_corpus,text):
    
    if 'total_count' in word_corpus:
        print("ERROR: keyword collision")
        print('    "total_count" is used as a dictionary key and appears in the word corpus')
        return 'NaN'
    
    count_dict = {}
    
    total_count = 0
    for word in word_corpus:
        word_count = len(re.findall(word,text))
        count_dict[word] = word_count
        total_count += word_count
        
    count_dict['total_count'] = total_count
    
    return count_dict


def count_past_present_future(text):
    
    """
    JEM Note: Copied from Larry's Conversations.py file'
    """
    if type(text) == str:
        text = text.split(' ')
    
    future_count = []
    present_count = []
    past_count = []
    
    # Temporal Reference
    tags = nltk.pos_tag(text)
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
            
    future_count = len(future_count)
    present_count = len(present_count)
    past_count = len(past_count)

    return future_count,present_count,past_count




def notched_range(data):
    mean = data.mean()
    median = np.median(data)
    upper,lower = np.percentile(data,[75,25])
    IQR = upper-lower
    n = data.shape[0]
    
    delta = 1.7*(1.25*IQR/(1.35*np.sqrt(n)))
    
    notch_range = [median-delta,median+delta]
    
    return notch_range,median,delta


def compare_distributions_kruskal(name1,name2,set1,set2,significance):
    no_significance = True
    try:
        stat,p = stats.kruskal(set1,set2,nan_policy = 'omit')
        if p<significance:
            print('  {}/{} =>  stat={}, p={}'.format(name1,name2,stat,p))
            no_significance = False
    except:
        donothing=1
        
    return no_significance



def calc_standard_error(counts,total,z):
    p = counts/total
    se = z*np.sqrt(p*(1-p)/total)
    return se
    
    
















