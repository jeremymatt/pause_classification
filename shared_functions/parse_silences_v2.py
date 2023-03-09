#general libraries
import re
import copy

from collections import defaultdict
import setupPCCRIconversations as stup


#functions

def getAlltags(conversations):
    all_tags = set()
    for conv in conversations.convs:
        for i in range(0,len(conv.turns)):
            turn = conv.turns[i]
            turn_sentence = ' '.join(word for word in turn)
            tags = re.findall('\<.*?\>', turn_sentence)
            for tag in tags:
                all_tags.add(tag)

    return all_tags




def tag_loc(temp_turn,tag):
    counter = 0
    for word in temp_turn:
        if tag in word:
            return counter
        
        counter = counter+1
    return counter
def remove_tags(temp_turn,all_tags):
    turn = []
    for word in temp_turn:
        if word not in all_tags:
            turn.append(word)
            
    return turn



##create manual metadata
silence_types = ['<s0>','<s1>','<s2>','<s3>','<s4>','<s5>','<s6>','<s7>','<s8>','<s9>']
#silence_types = ['<s1>']
silence_start = ['<ss>','<ss1>','<ss2>','<ss3>','<ss4>','<ss5>','<ss6>','<ss7>','<ss8>','<ss9>']
#silence_start = ['<ss5>']
silence_end = ['<se>','<se1>','<se2>','<se3>','<se4>','<se5>','<se6>','<se7>','<se8>','<ss9>']  


def find_tags(tags, turn_sentence):
    for tag in tags:
        if turn_sentence.find(tag) > -1:
            return 1
    return 0








def extract_silence (conv, turn_num_start, word_num_start, turn_temp, silence_nesting, silence_counts,silence_types,silence_end,num_sentences):
    d = {}
    d['Conversation'] = conv.key['Conversation']
    d['Patient'] = conv.key['Patient']
    d['silence start'] = re.findall("<(.*?)>", turn_temp[word_num_start])[0]
    d['pre silence'] = []
    d['post silence'] = []
    d['pre silence complete turn'] = []
    d['post silence complete turn'] = []
    d['pre silence n sentences'] = []
    d['post silence n sentences'] = []
    tag_num = d['silence start'][-1]

    if tag_num.isdigit():
        tag_num = int(tag_num)
        end_tag = '<se' + str(tag_num) + '>'
    else:
        tag_num = 0
        end_tag = '<se>'



    silence_nesting_pre = silence_counts
    word_num_start_temp = word_num_start
    pre_silence_end_turn = turn_num_start
    num_sentences_pre = num_sentences

    while silence_nesting_pre >= 0:

        for turn_num in range(turn_num_start, len(conv.turns)):
            if silence_nesting_pre < 0:
                break
            turn_current = copy.deepcopy(conv.turns[turn_num])

            # print(turn_sentence)
            for word_num in range(word_num_start_temp, len(turn_current)):
                if silence_nesting_pre < 0:
                    break

                if find_tags(silence_types, turn_current[word_num]) == 1:
                    silence_nesting_pre = silence_nesting_pre - 1
                    if silence_nesting_pre == -1:
                        d['silence type'] = re.findall("<(.*?)>", turn_current[word_num])[0]
                        pre_silence_end_word = word_num
                        pre_silence_end_turn = turn_num

            word_num_start_temp = 0


    if silence_nesting_pre >=0:
        print('Error: Silence end not found')
        return 0
    else:
        for i in range(turn_num_start, pre_silence_end_turn+1):
            turn_temp = copy.deepcopy(conv.turns[i])
            if i == pre_silence_end_turn:
                try:
                    end_point = pre_silence_end_word
                except:
                    print('ERROR - failed to assign end_point=pre_silence_end_word')
                    print(turn_temp)
                    print(d['Patient'])
            else:
                end_point = len(turn_temp)
            if i == turn_num_start:
                start_point = word_num_start
            else:
                start_point = 0
                
            try:
                d['pre silence'].extend(turn_temp[start_point:end_point])
            except:
                breakhere=1    
            d['pre silence complete turn'].extend(turn_temp[0:end_point])

            #pre silence end turn is the turn with <silence type> tag in it
            #pre silence end point is the word num with <silence tag in it>
            #turn num start is the <ss> tag
        for i in range(pre_silence_end_turn, 1,-1):

            if num_sentences_pre <=0:
                break
            turn_temp = copy.deepcopy(conv.turns[i])
            if i == pre_silence_end_turn:
                try:
                    end_point = pre_silence_end_word
                except:
                    print(turn_temp)
                    print(d['Patient'])
            else:
                end_point = len(turn_temp)
            for j in range(end_point-1,-1,-1):
                if ("." in turn_temp[j]):
                    num_sentences_pre = num_sentences_pre - 1

                d['pre silence n sentences'].insert(0,turn_temp[j])



    silence_nesting_post = silence_nesting
    word_num_start_temp = word_num_start
    silence_start_turn = turn_num_start
    while silence_nesting_post >= 0:

        for turn_num in range(silence_start_turn, len(conv.turns)):
            if silence_nesting_post <0:
                break
            turn_current = copy.deepcopy(conv.turns[turn_num])

            # print(turn_sentence)
            for word_num in range(word_num_start_temp, len(turn_current)):
                word = turn_current[word_num]
                if silence_nesting_post <0:
                    break
                if find_tags(silence_end, turn_current[word_num]) == 1:

                    silence_nesting_post = silence_nesting_post - 1
                    if silence_nesting_post == -1:
                        post_silence_end_word = word_num
                        post_silence_end_turn = turn_num

            word_num_start_temp = 0

    if silence_nesting_post >=0:
        print('Error: Post silence end not found')
        return 0
    else:
        for i in range(pre_silence_end_turn, post_silence_end_turn+1):
            turn_temp = copy.deepcopy(conv.turns[i])
            if i == post_silence_end_turn:
                end_point = post_silence_end_word
            else:
                end_point = len(turn_temp)
            if i == pre_silence_end_turn:

                start_point = pre_silence_end_word

            else:
                start_point = 0

            d['post silence'].extend(turn_temp[start_point:end_point])
            d['post silence complete turn'].extend(turn_temp[start_point:len(turn_temp)])

    return d


#



def get_silences(conversations, num_sentences = 0):
    silence_list = []
    conv_stats = defaultdict(lambda:[])
    total_turns = {}
    
    for conv in conversations.convs:
        print(conv.key['Patient'])
        silence_starts = 0
        silence_counts = 0
        silence_ends = 0
        silences = []
        con_sil_id = 0
        for turn_num in range(0, len(conv.turns)):
            turn_current = copy.deepcopy(conv.turns[turn_num])

            # print(turn_sentence)
            for word_num in range(0, len(turn_current)):
                if find_tags(silence_start, turn_current[word_num]) == 1:
                        d={}
                        word = turn_current[word_num]
                        silence_counts_temp = (silence_starts - silence_counts)
                        silence_nesting = silence_starts - silence_ends
                        # print('turn_num: {}'.format(turn_num))
                        # if turn_num == 57:
                        #     breakhere=1
                        silence = extract_silence(conv, turn_num, word_num, turn_current, silence_nesting, silence_counts_temp,silence_types, silence_end,num_sentences)
                        silence['silence_num'] = con_sil_id
                        con_sil_id+=1
                        silence_list.append(silence)
                        silence_starts = silence_starts + 1
                        
                        d['turn'] = turn_num
                        d['word_num']= word_num
                        d['silence type'] = silence['silence type']
                        silences.append(d)
                        
                if find_tags(silence_types, turn_current[word_num]) == 1:
                    word = turn_current[word_num]
                    silence_counts = silence_counts +1

                if find_tags(silence_end, turn_current[word_num]) == 1:
                    word = turn_current[word_num]
                    silence_ends = silence_ends +1
        
        p = conv.key['Patient']
        cNum = conv.key['Conversation']
        key = str(p)+'_'+str(cNum)
        total_turns[key] =len(conv.turns)
        
        conv_stats[key] = silences

    return silence_list,conv_stats,total_turns


#conversations = stup.PCCRI_setup({},transcripts_path = '../data/')
#silence_list = get_silences(conversations,num_sentences = 3)
#print('Total silences :'+str(len(silence_list)))

