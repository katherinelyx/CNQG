import pickle
import argparse
import numpy as np
#from IFTC import Options, IFTC
from CNQG import Options, HT
#import read_ht
import tensorflow as tf
import tqdm
from gensim.models import Word2Vec, KeyedVectors
import math


# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type = str, default = 'data/DD/train',
                    help = 'the directory to the train data')
parser.add_argument('--test_data_path', type = str, default = 'data/DD/test',
                    help = 'the directory to the test data')
parser.add_argument('--num_epochs', type = int, default = 20,
                    help = 'the number of epochs to train the data')
parser.add_argument('--batch_size', type = int, default = 64,
                    help = 'the batch size')
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'the learning rate')
parser.add_argument('--beam_width', type = int, default = 10,
                    help = 'the beam width when decoding')
parser.add_argument('--embedding_size', type = int, default = 512,
                    help = 'the size of word embeddings')
parser.add_argument('--embedding_path', type = str, default = 'GoogleNews-vectors-negative300.bin',#utils/sgns.weibo.word   GoogleNews-vectors-negative300.bin
                    help = 'the path of word embeddings file')
parser.add_argument('--num_hidden_units', type = int, default = 512,
                    help = 'the number of hidden units')
parser.add_argument('--save_path', type = str, default = 'model/',
                    help = 'the path to save the trained model to')
parser.add_argument('--restore_path', type = str, default = 'model/',
                    help = 'the path to restore the trained model')
parser.add_argument('--restore', type = bool, default = False,
                    help = 'whether to restore from a trained model')
parser.add_argument('--predict', type = bool, default = True,
                    help = 'whether to enter predicting mode')
parser.add_argument('--training', type = bool, default = False,
                    help = 'whether to enter training mode')
args = parser.parse_args()

def load_data(data_path):
    enc_x = np.load('{}/enc_x.npy'.format(data_path))
    context_lens = np.load('{}/context_lens.npy'.format(data_path))
    enc_x_lens = np.load('{}/enc_x_lens.npy'.format(data_path))
    
    dec_y = np.load('{}/dec_y.npy'.format(data_path))
    dec_y_lens = np.load('{}/dec_y_lens.npy'.format(data_path))
    
    x_topic = np.load('{}/x_topic.npy'.format(data_path))
    x_topic_lens = np.load('{}/x_topic_lens.npy'.format(data_path))
    transit_topic = np.load('{}/transit_topic.npy'.format(data_path))
    transit_topic_lens = np.load('{}/transit_topic_lens.npy'.format(data_path))
    cr_label = np.load('{}/cr_label.npy'.format(data_path))
    tr_label = np.load('{}/tr_label.npy'.format(data_path))
    q_type = np.load('{}/q_type.npy'.format(data_path))

    
    
    return enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, transit_topic, transit_topic_lens, cr_label, tr_label, q_type



if __name__ == '__main__':
    
    
    print('Load Data----------------')
    with open('{}/vocabulary.pickle'.format('data/DD'), 'rb') as file:
        vocabulary = pickle.load(file)
    with open('{}/vocabulary_reverse.pickle'.format('data/DD'), 'rb') as file:
        vocabulary_reverse = pickle.load(file)
    
    enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, \
        transit_topic, transit_topic_lens, cr_label, tr_label, q_type = load_data(args.train_data_path)
    max_dialog_len = enc_x.shape[1]
    max_utterance_len = enc_x.shape[2]
    max_topic_len = x_topic.shape[1]
    max_transit_len = transit_topic.shape[1]
    qword = ['what', 'how', 'when', 'which', 'who', 'where', 'why','do']
    
    qword_indx = []
    for word in qword:
        qword_indx.append(vocabulary[word])
    
    
    print('Set parametres------------------')
    options = Options(reverse_vocabulary = vocabulary_reverse,num_epochs = args.num_epochs,
                      batch_size = args.batch_size,
                      learning_rate = 0.001,
                      lr_decay = 0.995,
                      max_grad_norm = 5.0,
                      beam_width = args.beam_width,
                      vocabulary_size = len(vocabulary),                      
                      embedding_size = args.embedding_size,
                      num_hidden_units = args.num_hidden_units,
                      max_dialog_len = max_dialog_len,
                      max_utterance_len = max_utterance_len,
                      max_topic_len = max_topic_len,
                      max_transit_len = max_transit_len,
                      go_index = vocabulary['<go>'],
                      eos_index = vocabulary['<eos>'],
                      training_mode = args.training,
                      save_model_path = args.save_path,
                      dev_data_path = args.test_data_path,
                      recall_lambda = 1.0,
                      transit_lambda = 1.0,
                      qtype_lambda = 1.0,
                      lambda_decay = 0.5)  
       

    model = HT(options)

    if args.predict:
        model.restore(args.restore_path)
        predicted = model.predict(args.test_data_path, qword_indx)

        with open('result/CNQG.txt', 'w', encoding= 'utf-8') as f:
            for item in predicted: 
                f.write('Context: ')
                f.write(item['Context'])
                f.write('\n')
                f.write('Recall-Topic:')
                f.write(item['Recall'])
                f.write('\n')
                f.write('Transit-Topic:')
                f.write(item['Transit'])
                f.write('\n')
                f.write('Qword:')
                f.write(item['Qword'])
                f.write('\n')    
                f.write('Target: ')
                f.write(item['Target'])
                f.write('\n')
                f.write('Predict: ')
                f.write(item['Predict'])
                f.write('\n\n')
        
    else:
        if args.restore:
            model.restore(args.restore_path)
        else:
            model.init_tf_vars()
        model.dynamic_train(enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, transit_topic, transit_topic_lens, cr_label, tr_label, q_type, qword_indx)
        
