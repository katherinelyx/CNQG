import os
import pickle
import argparse
import numpy as np


def read_txt(path, data_name, mode):
    '''
    read data from txt file
    '''
    data_path = '{}/{}_{}.txt'.format(path, mode, data_name)
    data = []
    with open(data_path, 'r', encoding = 'utf-8') as f:
        txt = f.read().split('\n')
        txt = txt[:-1]
        if data_name == 'context':
            for line in txt:
                utts = []
                if ' || ' in line:
                    items = line.split(' || ')
                    for item in items:
                        utts.append(item.split(' '))
                elif ' ' in line and len(line)>2:
                    utts.append(line.split(' '))
                elif line in [' ','']:
                    utts.append(' ')
                else:
                    utts.append(line)
                data.append(utts)
        else:
            for line in txt:
                if ' ' in line and len(line)>1:
                    data.append(line.split(' '))
                elif line in [' ','']:
                    data.append(' ')
                else:
                    data.append(line)
    return data



def convert_to_integer_representation(contexts, responses, con_topics, transit_topics, CR_label, TR_label, patterns, vocabulary, max_dialog_len, max_utterance_len, max_topic_len, max_transit_len):
    
    assert len(contexts)==len(responses)==len(con_topics)
    # context
    enc_x = np.zeros((len(contexts), max_dialog_len+1, max_utterance_len+1), dtype = np.int32) # 0 for padding
    context_lens = np.array([len(context) for context in contexts], dtype = np.int32)
    enc_x_lens = np.zeros((len(contexts), max_dialog_len+1), dtype = np.int32)
        
    # response
    dec_y = np.zeros((len(responses), max_utterance_len + 1), dtype = np.int32) # placeholder for <EOS>
    dec_y_lens = np.array([len(response)+1 for response in responses], dtype= np.int32)

    # context topic
    x_topic = np.zeros((len(con_topics), max_topic_len), dtype = np.int32)
    x_topic_lens = np.array([len(xt) for xt in con_topics], dtype= np.int32)
    cr_label = np.zeros((len(CR_label), max_topic_len), dtype = np.int32)
    
    # transition topic
    transit_topic = np.zeros((len(transit_topics), max_transit_len), dtype = np.int32)
    transit_topic_lens = np.array([len(tt) for tt in transit_topics], dtype= np.int32)
    tr_label = np.zeros((len(TR_label), max_transit_len), dtype = np.int32)
    
    # pattern
    q_type = np.zeros((len(patterns), 8), dtype = np.int32)

    for j in range(len(contexts)):
        for k in range(len(contexts[j])):
            con_utt = []
            for word in contexts[j][k]:
                if word in vocabulary:
                    con_utt.append(vocabulary[word])
                else:
                    con_utt.append(vocabulary['<unk>'])
            if len(con_utt) > max_utterance_len:
                con_utt = con_utt[(len(con_utt)-max_utterance_len):]
            enc_x[j,k,:len(con_utt)] = con_utt
            enc_x_lens[j,k] = len(con_utt)
    
    for i in range(len(responses)):
        re_utt = []
        for word in responses[i]:
            if word in vocabulary:
                re_utt.append(vocabulary[word])
            else:
                re_utt.append(vocabulary['<unk>'])
        if len(re_utt) > max_utterance_len:
            re_utt = re_utt[:max_utterance_len]
        re_utt = re_utt + [vocabulary['<eos>']]
        dec_y[i,:len(re_utt)] = re_utt
        dec_y_lens[i] = len(re_utt)  

    for i in range(len(con_topics)):
        c_t = []
        if con_topics[i] != '' and con_topics[i] != ' ':            
            for word in con_topics[i]:
                if word in vocabulary:
                    c_t.append(vocabulary[word])
                else:
                    c_t.append(vocabulary['<unk>'])
            if len(c_t) > max_topic_len:
                c_t = c_t[:max_topic_len]
            x_topic[i,:len(c_t)] = c_t
            x_topic_lens[i] = len(c_t)
    
    for i in range(len(transit_topics)):
        t_t = []
        if transit_topics[i] != '' and transit_topics[i] != ' ':
            for word in transit_topics[i]:
                if word in vocabulary:
                    t_t.append(vocabulary[word])
                else:
                    t_t.append(vocabulary['<unk>'])
            if len(t_t) > max_transit_len:
                t_t = t_t[:max_transit_len]
            transit_topic[i,:len(t_t)] = t_t
            transit_topic_lens[i] = len(t_t)


    for i in range(len(CR_label)):
        crl = CR_label[i]
        if crl != '' and crl != ' ':
            if len(crl) > max_topic_len:
                crl = crl[:max_topic_len]
            cr_label[i, :len(crl)] = crl
    
    for i in range(len(TR_label)):
        trl = TR_label[i]
        if trl != ' ':
            if len(trl) > max_transit_len:
                trl = trl[:max_transit_len]
            tr_label[i, :len(trl)] = trl

    for i in range(len(patterns)):
        p = patterns[i]
        if p == 'what': q_type[i, 0] = 1
        elif p == 'how': q_type[i, 1] =1
        elif p == 'when': q_type[i, 2] = 1
        elif p == 'which': q_type[i, 3] = 1
        elif p == 'who': q_type[i, 4] = 1
        elif p == 'where': q_type[i, 5] = 1
        elif p == 'why': q_type[i, 6] = 1
        elif p == 'yes/no': q_type[i, 7] = 1
        
        

    return enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, transit_topic, transit_topic_lens, cr_label, tr_label, q_type


def save_data(enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, transit_topic, transit_topic_lens, cr_label, tr_label, q_type, save_path):
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save('{}/enc_x.npy'.format(save_path), enc_x)
    np.save('{}/context_lens.npy'.format(save_path), context_lens)
    np.save('{}/enc_x_lens.npy'.format(save_path), enc_x_lens)
    
    np.save('{}/dec_y.npy'.format(save_path), dec_y)
    np.save('{}/dec_y_lens.npy'.format(save_path), dec_y_lens)

    np.save('{}/x_topic.npy'.format(save_path), x_topic)
    np.save('{}/x_topic_lens.npy'.format(save_path), x_topic_lens)

    np.save('{}/transit_topic.npy'.format(save_path), transit_topic)
    np.save('{}/transit_topic_lens.npy'.format(save_path), transit_topic_lens)
    
    np.save('{}/cr_label.npy'.format(save_path), cr_label)

    np.save('{}/tr_label.npy'.format(save_path), tr_label)

    np.save('{}/q_type.npy'.format(save_path), q_type)
    


if __name__ == '__main__':
    path = 'data/DD'
    
    # Load vocabulary
    with open('{}/vocabulary.pickle'.format(path), 'rb') as file:
        vocabulary = pickle.load(file)
    with open('{}/vocabulary_reverse.pickle'.format(path), 'rb') as file:
        vocabulary_reverse = pickle.load(file)
    
    
    # Load data    
    # Training
    mode = 'train'
    train_contexts = read_txt(path, 'context', mode)
    train_responses = read_txt(path, 'response', mode)
    train_context_topics = read_txt(path, 'context_topics', mode)
    train_response_topics = read_txt(path, 'response_topics', mode)
    train_transition_topics = read_txt(path, 'transition_topics', mode)
    train_CR_label = read_txt(path, 'CR_label', mode)
    train_TR_label = read_txt(path, 'TR_label', mode)
    train_pattern = read_txt(path, 'pattern', mode)
    # Testing
    mode = 'test'
    test_contexts = read_txt(path, 'context', mode)
    test_responses = read_txt(path, 'response', mode)
    test_context_topics = read_txt(path, 'context_topics', mode)
    test_response_topics = read_txt(path, 'response_topics', mode)
    test_transition_topics = read_txt(path, 'transition_topics', mode)
    test_CR_label = read_txt(path, 'CR_label', mode)
    test_TR_label = read_txt(path, 'TR_label', mode)
    test_pattern = read_txt(path, 'pattern', mode)
    

    
    # Convert to index representation
    max_dialog_len = 15
    max_utterance_len = 50
    max_topic_len = 50
    max_transit_len = 50
    
    train_enc_x, train_context_lens, train_enc_x_lens, train_dec_y, train_dec_y_lens, train_x_topic, train_x_topic_lens, train_transit_topic, \
        train_transit_topic_lens, train_cr_label, train_tr_label, train_qtype = convert_to_integer_representation(train_contexts, train_responses, \
            train_context_topics, train_transition_topics, train_CR_label, train_TR_label, train_pattern, vocabulary, max_dialog_len, max_utterance_len, \
                max_topic_len, max_transit_len)
    test_enc_x, test_context_lens, test_enc_x_lens, test_dec_y, test_dec_y_lens, test_x_topic, test_x_topic_lens, test_transit_topic, \
        test_transit_topic_lens, test_cr_label, test_tr_label, test_qtype = convert_to_integer_representation(test_contexts, test_responses, \
            test_context_topics, test_transition_topics, test_CR_label, test_TR_label, test_pattern, vocabulary, max_dialog_len, max_utterance_len, \
                max_topic_len, max_transit_len)
    

    # Save to pickle
    save_data(train_enc_x, train_context_lens, train_enc_x_lens, train_dec_y, train_dec_y_lens, train_x_topic, train_x_topic_lens, train_transit_topic, \
        train_transit_topic_lens, train_cr_label, train_tr_label, train_qtype, path + '/train')
    save_data(test_enc_x, test_context_lens, test_enc_x_lens, test_dec_y, test_dec_y_lens, test_x_topic, test_x_topic_lens, test_transit_topic, \
        test_transit_topic_lens, test_cr_label, test_tr_label, test_qtype,path + '/test')


   