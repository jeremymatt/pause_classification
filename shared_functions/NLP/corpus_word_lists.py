# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:55:06 2023

@author: jerem
"""

"""
These word corpuses assume that regular expressions are used to find word
instances.  NOTES:
    1. use ".??" to match any one character or no character (e.g., don't vs dont)
    2. use '\\b' to match word boundaries.  
        (e.g., "\\brisk" would match "risk", "risks", "risked", etc but 
         WOULD NOT match "brisk")
    
"""

def getSymptomWords():
    symptomWords = ['comfortable', 'worried', '\\btired\\b', 'painful', 'symptom', 
                     'shortness', 'hurting', 'confused', 'uncomfortable', 'weak', 
                     'happy', 'comfort', 'sleepy', 'depressed', '\\bhurts\\b', 'symptoms', 
                     '\\bpain\\b', 'breathing', 'cough', 'constipation', '\\bdry\\b', 'energy', 
                     'appetite', 'awake', '\\bhurt\\b', 'coughing', 'sleep', 'breathe', 
                     'strength', 'breath', 'sleeping', 'bothering', 'nausea', 
                     'strong', 'anxiety', '\\bwake\\b', 'scary', 'depression', 'worry', 
                     'stronger', 'anxious']
    
    return symptomWords

def getTreatmentWords():
    treatmentWords = ['morphine', 'patch', 'medications', '\\bdrug', '\\btrial', 'CPR', 
                      '\\bline', 'Tylenol', 'button', 'doses', '\\bdrugs', 'medical', 
                      'feeding', 'oxygen', 'Ativan', 'Oxycodone', 'therapy', 
                      'Dilaudid', 'chemotherapy', 'machine', 'antibiotics', 
                      'treatment', 'radiation', 'surgery', 'treat', 'dose', 
                      'meds', 'medicines', 'fluids', 'tube', 'hospice', 'medicine', 
                      'dialysis', 'methadone', '\\boral\\b', 'ventilator', 'milligrams', 
                      'management', 'resuscitation', 'fentanyl', 'chemo', '\\bpill\\b', 
                      'nutrition', '\\bICU\\b', 'milligram', 'medication', 'procedure', 
                      'liquid', 'treatments', '\\bIV\\b', '\\bpills\\b']
    
    return treatmentWords
    
def getStopWords():
    
    """
    NOTE NOTE NOTE:
        UPDATE THIS CORPUS TO BE CONSISTENT WITH REGULAR EXPRESSIONS IF 
        THIS IS USED WITH THE UPDATE CORPUS COUNTING METHOD
    """
    # This stop word list was downloaded/generated from the NLTK package on 9/11/19
    stop_words = ['i', 'me', 'my', 'myself' 'we', 'our', 'ours', 'ourselves',
                  'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his' 'himself', 'she',
                  "she's", 'her', 'hers', 'herself', 'it', "it's", 'its' 'itself',
                  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                  'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                  'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                  'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                  'into', 'through', 'during', 'before', 'after', 'above', 'below',
                  'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                  'under', 'again', 'further', 'then', 'once', 'here', 'there',
                  'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                  'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                  'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                  't', 'can', 'will','just', 'don', "don't", 'should', "should've",
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                  "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                  "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                  'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                  'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                  'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
                  "wouldn't"]
    
    return stop_words
    
def getPossibilityWords():
    # Strong subgroup of probability/possibility indicators
    possiblity_words = ['could','chance','changeable','generally','feasible',
                              'likely','\\bmay\\b','maybe','might','nearly','perchance',
                              'perhaps','plausibly','possibility','possible',
                              'potential','probability','probable','reasonable']
    return possiblity_words

def getQualifierWords():
    # Uncertainty subgroup of qualifiers
    uqualifier_words = ['allegedly','approximately','relatively','roughly',
                              'slightly','usually','virtually']
    return uqualifier_words

def getPrognosisWords():
    prog_words = ['chance','chances','feasible','feasibly','probably','probable',
                  '\\bmay\\b','maybe','might','likely','unlikely','likelihood','plausible',
                  'plausibly','possibility','possible','impossible','potentially',
                  'potential','probability','percent','certain','uncertain',
                  'expect','expectation','unpredictable','predictable','anticipate',
                  'prognosticate','predict','prediction','\\brisk','speculate','suspect',
                  'postulate','\\bhope','foresee','estimate','guess']
    return prog_words

def getHedgingWords():
    hedging_words = ['allude','anticipate','assess','consider','contemplate',
                     'deem','doubt','estimate','expect','foresee','guess','\\bhint',
                     '\\bhope','hypothesize','imagine','\\bimply','infer','misinterpret',
                     'misjudge','perceive','ponder','postulate','predict','presume',
                     'presuppose','prognosticate','\\brisk','\\bseem','speculate',
                     'suggest','suppose','theorize','think','worry']
    return hedging_words

def getConfusionWords():
    confusion_words = ['ambiguity','baffled','befuddle','befuddled','bewilder',
                       'complex','complicated','confound','confuse','confused',
                       'convoluted','discombobulate','dubious','dumbfound',
                       'incertain','inconstant','mistrust','misunderstand',
                       'perplex','perplexed','uncertain','unclear','unconvinced',
                       'undecided','unexpected','unlikely','unpredictable','unsure']
    return confusion_words

def getStrongConfusionWords():
    
    """
    notes:
        1. use ".??" to match any one character or no character (e.g., don't vs dont)
        2. use '\\b' to match word boundaries
    """
    confusion_words = ['\\ba bit\\b','contemplate','\\bhope\\b','most likely','presume',
                       'take a chance','\\ba little','convoluted','hypothes',
                       'most of the time','presuppose','theor','allegedly',
                       'could','imagine','nearly','probability','think',
                       'allude to\\b','curiosity','\\bimpl','not certain',
                       'probabl','\\btry\\b','ambiguity','\\bdeem\\b',
                       'in all likelihood','not convinced','prognosticate',
                       'uncertain','anticipate','discombobulate',
                       'in all probability','not know','\\bquite\\b','unclear',
                       'approximately','do not understand',
                       'incertain','not sure','reasonable','unconvinced','\\bassess',
                       'don.??t know','inconstant','\\boften','relatively',
                       'undecided','baffle','don.??t understand','infer',
                       'perceive','\\brisk','unexpected','befuddle','doubt',
                       'kind of','perchance','roughly','unlikely','bewilder',
                       'dubious','likel','perhaps','\\bseem\\b','unpredictable',
                       'call into question','dumbfound','\\bmay\\b','perplex',
                       'should','unsure','chance','estimate','maybe','plausibl',
                       'slightly','usually','changeable','expect','might',
                       'ponder','somewhat','\\bvary\\b','complex','feasibl',
                       'misinterpret','possibility','sort of\\b','virtually',
                       'complicate','foresee','misjudge','possibl','speculate',
                       'whether','confound','generally','mistrust','postulate',
                       'suggest',',whether or not','confus','guess',
                       'misunderstand','potential','suppose','worr',
                       'consider','hint','mixed up','predict','suspect']
    
    confusion_words = [word.lower() for word in confusion_words]
    
    
    return confusion_words


def getFirstPersonSingular():
    regex_list = ['\\bi\\b','\\bi.??ll\\b','\\bme\\b','\\bmy\\b','\\bmine\\b','\\bmyself\\b']

    return regex_list


def getFirstPersonPlural():
    regex_list = ['\\bwe\\b','\\bus\\b','\\bour\\b','\\bours\\b','\\bourselves\\b']

    return regex_list


def get2nd3rdPerson():
    regex_list = ['\\byou\\b','\\byour\\b','\\byours\\b','\\byourself\\b','\\byourselves\\b',
                        '\\bhe\\b','\\bhim\\b','\\bhis\\b','\\bhimself\\b','\\bshe\\b','\\bher\\b',
                        '\\bhers\\b','\\bherself\\b','\\bit\\b','\\bits\\b','\\bitself\\b','\\bthey\\b',
                        '\\bthem\\b','\\btheir\\b','\\btheirs\\b','\\bthemself\\b','\\bthemselves\\b']
    
    return regex_list


def getStrongLonelinessWords():
    
    
    """
    Notes:
        0. Must remove all characters that are not a letter or a space
        1. "associate" matches "associated" and "disassociated"
        2. "belong" matches "belonging"
        3. "brother" matches "brother-in-law"
        4. "care" matches "careful" and "careless"
        5. "child" matches "stepchild", "grandchild", and "children"
        6. "community" and "community" both match "church-community"
        7. "collaborat" matches "collaborate" and "collaborator"
        8. "companion" matches "companionship"
        9. "daughter" matches "granddaughter", "stepdaughter", "goddaughter", and "daughter-in-law"
        10. "desolat" matches "desolate" and "desolation"
        11. "embrace" matches "embraced"
        12. "estrange" matches "estranged"
        13. "father" matches "father-in-law"
        14. "friend" matches (girlfriend, boyfriend, friendly, unfriendly, friends, friendship)
        15. "mother" matches "mother-in_law"
        16. "neglect" matches "neglected"
        17. "nobody" and "company" both match "Nobody to keep me company"
        18. Not sure what "OT" is but leaving in anyway
        19. "sibling" matches "stepsibling" and "siblings"
        20. "sister" matches "stepsister" and "sister-in-law"
        21. "support" matches "supporter" and "unsupported"
        22. "together" matches "togetherness"
        23. "understood" matches "misunderstood"
        24. "unite" matches  (united, reunite, disunite, disunited)
    """
    loneliness_words = [
        'Abandoned','Affection','Alienated','Assistant','Associate','\\bAunt',
        'Belong','\\bBetray','Bleak','Brother','Brushed aside','Brushed off',
        'Buddy','By my side','By myself','\\bCare','uncared for','Cast away',
        '\\bCat','Cherished','Child','\\bChum','Church','\\bClub','Cohort',
        'Coldshoulder','Cold shoulder','Collaborat','Colleague','Community',
        'Companion','Company','Compassionate','Comrade','Confidante',
        'Confide in','Correspondence','Count on','Cousin','Crying out',
        'Cuddling','\\bDad','Daughter','Dejected','Demoralized','Desolate',
        'Disowned','Disrespected','\\bDog','Don.??t have someone',
        'Drifted apart','Embrace','Estrange','Family','Father','\\bFoe\\b','\\bFolk\\b',
        'Friend','Get together','Get with','\\bGod\\b','Has my back','Heard from',
        'Husband','Ignored','In the know','In touch','Inseparable','Intimacy',
        'Invited','Jesus','Kept apart','Kept in the dark','\\bKid\\b','\\bkids\\b',
        'stepkid','grandkid','\\bKiss','Lean on','Left behind',
        'Left high and dry','Left out','Listened to','\\bLone\\b','\\blonely\\b',
        '\\balone\\b','\\blonesome\\b','\\bLost\\b','\\bLove\\b','\\bloved\\b','truelove','unloved',
        '\\bMA\\b','Married','\\bMate','Mentor','Misfit','Missing out','\\bMom\\b','\\bmum\\b',
        'Mother','Neglect','Neighbor','Nephew','Niece','No one','Nobody',
        'Not a part of something','Not included','Not listened to','On my own',
        'Ostracized','\\bOT\\b','Out of place','\\bPal\\b','\\bpals\\b','\\bParent',
        'Participation','Partner','Relative','Rely on','Respected','Ridiculed',
        'Roommate','Scorned','\\bShare\\b','\\bshared\\b','\\bunshared\\b','\\bShun\\b','\\bshunned\\b',
        'Shut out','Sibling','Sidekick','Sister','Solitary','Solitude','\\bSon\\b','\\bSons\\b',
        'stepson','godson','grandson','son-in-law','Spouse','Support','Teacher',
        '\\bTeam\\b','\\bteammate\\b','There for me','Together','Touch base with',
        'Unaccompanied','Uncaring','Uncle','Understood','Unite','Unaccompanied',
        'Visit','\\bWe\\b','\\bWife\\b','With me','With others','Withdrawn']
    
    loneliness_words = [word.lower() for word in loneliness_words]
    
    return loneliness_words
    