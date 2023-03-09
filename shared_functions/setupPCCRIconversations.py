import numpy as np
import pandas as pd
from Conversation import PCCRI_Conversation, Primary_PCCRI_Conversation
from Conversations import Conversations
import os

def PCCRI_setup(convOpt ={} ,transcripts_path = '../../Research_Latest/transcripts_clean/all/'):
    if len(convOpt.keys())== 0:
        convOpt = {'merge':False,
                   'primary':False, # (updated in conditional statement below when True)
                   'stop_words':[],  # nlp.getStopWords()
                   'rem_anomolies':True,
                   'silence_features':True,
                   'enviro_features':True,
                   'emotions_features':True,
                   'remove_nonAlphabets': False
                   }
    
    #path = '../../Research_Latest/transcripts_clean/'
    #path = '../data/transcripts with silence tags parsing/'
    path = transcripts_path
    main_convo_filename = 'main_convo_data.csv' # Main/Primary Conversations
    #main_convo_filename = 'data/main_convo_data.csv' # Main/Primary Conversations
    hu_filename = '../data/heard_understood.xlsx' # Heard/understood scores
    # main_convo_filename = None # Main/Primary Conversations
    #main_convo_filename = 'data/main_convo_data.csv' # Main/Primary Conversations
    hu_filename = None # Heard/understood scores
    
    #files = os.listdir('all')
    #files = os.listdir('../../Research_Latest/transcripts_clean/all')
    files = os.listdir(transcripts_path)
    folder = transcripts_path
    #folder = path+'all/'
    
    if hu_filename == None:
        f = files[0]
        pid = list(set([f[:4] for f in files if f[:4].isnumeric()]))
        
        hu_scores = np.zeros([len(pid),3])
        hu_scores[:,0] = pid
        hu_scores = hu_scores.astype(int)
    else:
        hu_scores = np.array(pd.read_excel(hu_filename)) # Heard/understood scores
    
    if main_convo_filename == None:
        files = [file for file in files if os.path.isfile(os.path.join(transcripts_path,file))]
        breakhere=1
        pid = list([f[:4] for f in files if f[5]=='A'])
        cid = [ord(f[5])-ord('A')+1 for f in files if f[5]=='A']
        pidcid = list(zip(pid,cid))
        mainConvo = np.array(pidcid).astype(int)
    else:
        mainConvo = np.genfromtxt(main_convo_filename,delimiter=',') # col 0: pid; col 1: cid
    
    convs = Conversations([]) # Initialize list of conversations
    
    # Loop through files
    for filename in files:
        # If valid filename (i.e., transcript)
        if (filename[-3:]=='txt') and (filename[0] == '0'):
            
            pid = int(filename[:4]) # Patient ID
            
            conv_ID = filename[5]
            breakhere=1
            if (ord(conv_ID)>=48) and (ord(conv_ID)<=57):
                cid = int(conv_ID)
            else:
                cid = ord(conv_ID)-ord('A')+1 # Conversation Number
            
            hu_mask = hu_scores[:,0]==pid # Index of heard/understood score
            
            # If a Primary Conversation with before AND after H/U scores
            # if np.any(np.all([pid,cid] == mainConvo,1)) and not np.any(hu_scores[hu_mask,1:][0]==99) and not np.any(np.isnan(hu_scores[hu_mask,1:][0])):            
            if False:
                convOpt['primary'] = True
                conv = Primary_PCCRI_Conversation(folder, filename, convOpt, hu_scores[hu_mask,1][0], hu_scores[hu_mask,2][0])
                convs.addConv(conv)
    
            else:
                convOpt['primary'] = False
                conv = PCCRI_Conversation(folder, filename, convOpt)
                convs.addConv(conv)
                
    # Parallel list of patient/convsersations to remove from dataset:
    remConvs = [[184,2], # Only 1 patient turn (just clinician speaking)
                [600,2], # Patient responses written, so no perception of content
                [211,1], # Less than 20 turns, patient asks to turn off recorder
                [639,2]] # Less than 20 turns, most patient turns inaudible
    
    if convOpt['rem_anomolies']:
        for conv in remConvs:
            convs.removeConv(convs.getConv(conv[0],conv[1]))
    '''
    # Add duration feature (in sec)
    file_name = 'data/conversation_durations_ALL.csv' # data file name
    descrip_file = 'data/duration_feature_description.xlsx'
    pid = 'patient_num' # Column header for patient ID
    cid = 'conv_num' # Column header for clinician number
    ignore_cols = ['file_name'] # mp3 file name, not needed, note we input a list here
    convs.import_features_PCCRI(file_name, pid, cid, ignore_cols)
    convs.import_feature_descrip(descrip_file, ignore_cols)
    
    # Add Feature Sets
    # (I will eventually use a function/loop to add each set instead of hard-coding)
    
    if convOpt['silence_features']:
        file_name = 'data/silences_by_conversation.xlsx' # data file name
        descrip_file = 'data/silences_features_descrip.xls' # feature descriptions
        pid = 'pid' # Column header for patient ID
        cid = 'cid' # Column header for clinician number
        convs.import_features_PCCRI(file_name, pid, cid)
        convs.import_feature_descrip(descrip_file)

    if convOpt['enviro_features']:
        file_name = 'data/environmental_features.xlsx'
        descrip_file = 'data/environmental_feature_descriptions.xlsx'
        pid = 'p0_patient_id' # Column header for patient ID
        cid = 'convo_num' # Column header for clinician number
        convs.import_features_PCCRI(file_name, pid, cid)
        convs.import_feature_descrip(descrip_file)
        
    if convOpt['emotions_features']:
        file_name = 'data/emotions_volley_features.xlsx'
        descrip_file = 'data/emotion_volley_features_description.xlsx'
        pid = 'p0_patient_id' # Column header for patient ID
        cid = 'convo_num' # Column header for clinician number
        convs.import_features_PCCRI(file_name, pid, cid)
        convs.import_feature_descrip(descrip_file)
    '''
    return convs

"""
# Not quite ready to use this yet, for now hard-codinf each feature set
def __import_features_and_descriptions__(file_name, descrip_file, pid, cid):
    file_name = 'emotions_volley_features.xlsx'
    descrip_file = 'emotion_volley_features_description.xlsx'
    pid = 'p0_patient_id' # Column header for patient ID
    cid = 'convo_num' # Column header for clinician number
    convs.import_features_PCCRI(file_name, pid, cid)
    convs.import_feature_descrip(descrip_file)
"""