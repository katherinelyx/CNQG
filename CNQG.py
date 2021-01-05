import numpy as np
import tensorflow as tf
import nltk
import math
import transformer
from tf2_ndg_benckmarks.metrics.distinct import Distinct



def w2f(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        for line in data:
            f.write(str(line))
            f.write('\n')
        f.close()


def read_data(data_path):
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


class Options(object):
    '''Parameters used by the HierarchicalSeq2Seq model.'''
    def __init__(self, reverse_vocabulary, num_epochs, batch_size, learning_rate, lr_decay, max_grad_norm, beam_width, vocabulary_size,embedding_size,
                 num_hidden_units, max_dialog_len, max_utterance_len, max_topic_len, max_transit_len, go_index, eos_index, training_mode, 
                 save_model_path, dev_data_path, recall_lambda, transit_lambda, qtype_lambda, lambda_decay):
        super(Options, self).__init__()

        self.reverse_vocabulary = reverse_vocabulary
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.max_grad_norm = max_grad_norm
        self.beam_width = beam_width
        
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_hidden_units = num_hidden_units
        self.max_dialog_len = max_dialog_len
        self.max_utterance_len = max_utterance_len
        self.max_topic_len = max_topic_len
        self.max_transit_len = max_transit_len
        self.go_index = go_index
        self.eos_index = eos_index
        self.training_mode = training_mode
        
        self.save_model_path = save_model_path
        self.dev_data_path = dev_data_path

        self.recall_lambda = recall_lambda
        self.transit_lambda = transit_lambda
        self.qtype_lambda = qtype_lambda
        
        self.lambda_decay = lambda_decay
        
class HT(object):
    '''A hierarchical sequence to sequence model for multi-turn dialog generation.'''
    def __init__(self, options):
        super(HT, self).__init__()

        self.options = options

        self.build_graph()
        config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
        self.session = tf.Session(graph = self.graph)
        #self.writer = tf.summary.FileWriter("logs/", self.session.graph)

    def __del__(self):
        self.session.close()
        print('TensorFlow session is closed.')

    def build_graph(self):
        print('Building the TensorFlow graph...')
        opts = self.options

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('Input'):
                # turn, Sc, B
                self.context = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.max_utterance_len, opts.batch_size], name ='context')
                self.turn_num = tf.placeholder(tf.int32, shape = [opts.batch_size], name ='turn_num')
                self.context_lens = tf.placeholder(tf.int32, shape = [opts.max_dialog_len, opts.batch_size], name ='context_lens')

                # turn, St, B
                self.context_topic = tf.placeholder(tf.int32, shape= (None, opts.max_topic_len), name ='context_topic')
                self.con_topic_len = tf.placeholder(tf.int32, [opts.batch_size], name ='con_topic_len')

                self.transit_topic = tf.placeholder(tf.int32, [None, opts.max_transit_len], name ='transit_topic')
                self.transit_topic_len = tf.placeholder(tf.int32, [opts.batch_size], name ='transit_topic_len')

                self.response = tf.placeholder(tf.int32, [opts.batch_size, None], name ='response')
                self.response_len = tf.placeholder(tf.int32, [opts.batch_size], name ='response_len')

                self.cr_label = tf.placeholder(tf.float32, shape=(None, opts.max_topic_len),name='cr_label')
                self.tr_label = tf.placeholder(tf.float32, shape=(None, opts.max_transit_len),name='tr_label')

                self.qtype = tf.placeholder(tf.float32, [None, 8], name = 'question_type')
                self.qword = tf.placeholder(tf.int32, [None], name = 'question_word')


            with tf.variable_scope('embedding', reuse = tf.AUTO_REUSE):
                embeddings = tf.get_variable('lookup_table',
										dtype = tf.float32,
										shape = [opts.vocabulary_size, opts.embedding_size],
										initializer = tf.contrib.layers.xavier_initializer(),
                                        trainable= True)
                        
            
            # The Context Encoder.
            with tf.variable_scope('context_encoder', reuse = tf.AUTO_REUSE):
                # Perform encoding.
                con_outputs = []
                con_final_states = []
                # turn, S, B, D
                self.context = tf.reverse(self.context, [1])
                context_embed = transformer.embedding(self.context, opts.vocabulary_size, opts.embedding_size, reuse=tf.AUTO_REUSE)
                #context_embed = tf.nn.embedding_lookup(embeddings,self.context)
                for i in range(opts.max_dialog_len):
                    # S,B,D ---> B, S,D
                    in_seq = tf.transpose(context_embed[i,:,:,:], perm=[1, 0, 2])
                    # utt_st: B,S,E
                    with tf.variable_scope('enc_cell', reuse = tf.AUTO_REUSE):
                        con_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                    ((con_fw_outputs, con_bw_outputs), (con_fw_final_state, con_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=con_gru_cell, cell_bw=con_gru_cell, inputs=in_seq, sequence_length=self.context_lens[i,:], dtype=tf.float32)
                    con_out = tf.add(con_fw_outputs, con_bw_outputs)
                    con_st = tf.add(con_fw_final_state, con_bw_final_state)
                    # B                    
                    con_final_states.append(con_st)
                    con_outputs.append(con_out)
                print('==== The length of con_outputs: ', len(con_outputs))
                print('==== The Shape of con_out: ', con_outputs[0].get_shape())
                print('==== The length of con_final_states: ', len(con_final_states))
                print('==== The Shape of con_st: ', con_final_states[0].get_shape())

                # turn, B, D
                con_rnn_input = tf.reshape(tf.stack(con_final_states), [opts.max_dialog_len, opts.batch_size, opts.num_hidden_units])
                # B, turn, D
                con_rnn_input = tf.transpose(con_rnn_input, perm=[1, 0, 2])
                with tf.variable_scope('context_cell', reuse = tf.AUTO_REUSE):
                    rnn_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                    #rnn_gru_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(opts.num_hidden_units, reuse=tf.AUTO_REUSE) for _ in range(opts.num_hidden_layers)])
                rnn_output, rnn_state = tf.nn.dynamic_rnn(cell = rnn_gru_cell, inputs = con_rnn_input, sequence_length = self.turn_num,dtype = tf.float32)
                
                print('==== The Shape of rnn_state: ', rnn_state.get_shape())
                print('==== The Shape of rnn_output: ', rnn_output.get_shape())

            
            def focal_loss(select, mlp_output,alpha=0.25, gamma=2):
                
                sigmoid_p = tf.nn.sigmoid(mlp_output)
                
                #zeros = tf.zeros_like(sigmoid_p, dtype = sigmoid_p.dtype)
                zeros = tf.zeros_like(sigmoid_p, dtype = sigmoid_p.dtype)
                pos_p_sub = tf.where(select > zeros, select-sigmoid_p, zeros)
                neg_p_sub = tf.where(select > zeros, zeros, sigmoid_p)
                cross_entropy = -alpha * (pos_p_sub**gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-08, 1.0))\
                    -(1-alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0-sigmoid_p, 1e-08, 1.0))
                return tf.reduce_mean(cross_entropy)
            
            
            with tf.variable_scope('recall_module', reuse = tf.AUTO_REUSE):
                
                def multilayer_perceptron(x, weight, bias):

                    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
                    layer1 = tf.nn.relu(layer1)
                    layer2 = tf.add(tf.matmul(layer1, weight['h2']), bias['h2'])
                    layer2 = tf.nn.relu(layer2)
                    layer3 = tf.add(tf.matmul(layer2, weight['h3']), bias['h3'])
                    layer3 = tf.nn.relu(layer3)
                    out_layer = tf.add(tf.matmul(layer3, weight['out']), bias['out'])
                    return out_layer

                ### MLP parameters
                weight = {
                    'h1': tf.Variable(tf.random_normal([opts.num_hidden_units, 1024]),name='MLP/weight_h1'),
                    'h2': tf.Variable(tf.random_normal([1024, 512]),name='MLP/weights_h2'), 
                    'h3': tf.Variable(tf.random_normal([512, 128]),name='MLP/weights_h3'), 
                    'out': tf.Variable(tf.random_normal([128, opts.max_topic_len]),name='MLP/weights_out')                    
                }
                bias = {
                    'h1': tf.Variable(tf.random_normal([1024]),name='MLP/bias_h1'),
                    'h2': tf.Variable(tf.random_normal([512]),name='MLP/bias_h2'), 
                    'h3': tf.Variable(tf.random_normal([128]),name='MLP/bias_h3'), 
                    'out': tf.Variable(tf.random_normal([opts.max_topic_len]),name='MLP/bias_out')
                }

                self.recall_input = rnn_state
                self.recall_output = multilayer_perceptron(self.recall_input,weight,bias)
                print('==================self.recall_output:', self.recall_output.get_shape())
                
                masked_value = tf.ones_like(self.recall_output) * (-math.pow(2, 32) + 1)
                print('==================masked_value:', masked_value.get_shape())
                recall_mask = tf.sequence_mask(self.con_topic_len, opts.max_topic_len)
                masked_recall_output = tf.where(recall_mask, self.recall_output, masked_value)
                print('==================masked_recall_output:', masked_recall_output.get_shape())                
                
                
                self.recall_loss = focal_loss(self.cr_label, self.recall_output)
                #self.MLP_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.select,logits = self.MLP_output))
                
                # B, label_len
                weightes = tf.nn.sigmoid(masked_recall_output)
                print('==================weightes:', weightes.get_shape())
                top_k_indx = tf.nn.top_k(weightes, 10)[1]
                print('==================top_k_indx:', top_k_indx.get_shape())
                top_k_select_topics = tf.batch_gather(self.context_topic, top_k_indx)
                print('==================top_k_select_topics:', top_k_select_topics.get_shape())
                self.select_context_topic = top_k_select_topics

                recall_topics_embed = tf.nn.embedding_lookup(embeddings,top_k_select_topics)
                #selected = tf.layers.dense(select_topics_embed, opts.num_hidden_units, activation = tf.nn.relu)
            


            with tf.variable_scope('transit_module', reuse = tf.AUTO_REUSE):
                
                relevance_layer = tf.layers.Dense(opts.num_hidden_units)
                transit_topic_embed = tf.nn.embedding_lookup(embeddings,self.transit_topic)# B, T, D
                t_seq_len = tf.shape(transit_topic_embed)[1]
                H = tf.tile(tf.expand_dims(rnn_state, 1), [1, t_seq_len, 1]) # B, D ---> B, T, D
                energy = tf.nn.tanh(relevance_layer(tf.concat([H, transit_topic_embed], 2))) # B, T, 2D---> B, T, D
                energy = tf.transpose(energy, [0, 2, 1]) # B, D, T
                v = tf.get_variable('v',dtype = tf.float32,shape = [opts.num_hidden_units],\
                    initializer = tf.random_normal_initializer(mean=0,stddev=1/np.sqrt(opts.num_hidden_units)), trainable= True)
                V = tf.expand_dims(tf.tile(tf.expand_dims(v, 0), [opts.batch_size, 1]), 1) # D--> 1, D --> B, D --> B, 1, D
                energy = tf.matmul(V, energy) # B, 1, T
                self.transit_score = tf.nn.sigmoid(tf.squeeze(energy, 1)) # B, 1,T--> B, T --sigmoid(B, T)
                print('==================self.transit_score:', self.transit_score.get_shape())
                
                t_masked_value = tf.ones_like(self.transit_score) * (-math.pow(2, 32) + 1)
                print('==================t_masked_value:', t_masked_value.get_shape())
                transit_mask = tf.sequence_mask(self.transit_topic_len, opts.max_transit_len)
                masked_transit_score = tf.where(transit_mask, self.transit_score, t_masked_value)
                print('==================masked_transit_score:', masked_transit_score.get_shape())        

                self.transit_loss = focal_loss(self.tr_label, self.transit_score)
                
                # B, label_len
                t_top_k_indx = tf.nn.top_k(masked_transit_score, 5)[1]
                print('==================t_top_k_indx:', t_top_k_indx.get_shape())
                top_k_select_transit = tf.batch_gather(self.transit_topic, t_top_k_indx)
                print('==================top_k_select_transit:', top_k_select_transit.get_shape())
                self.select_transit_topic = top_k_select_transit

                transit_topics_embed = tf.nn.embedding_lookup(embeddings,top_k_select_transit)
            

            with tf.variable_scope('pattern', reuse = tf.AUTO_REUSE):
                # Question Type
                qt_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                # defining initial state
                qt_initial_state = rnn_state
                # B,(Sc + Sr + St), D
                qt_input = tf.concat([rnn_output, recall_topics_embed, transit_topics_embed], 1)
                # 'state' is a tensor of shape [batch_size, cell_state_size]
                qt_outputs, qt_state = tf.nn.dynamic_rnn(qt_cell, qt_input,initial_state=qt_initial_state,dtype=tf.float32)
                qt_dense_layer = tf.layers.Dense(units = 7,kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1))
                qt_logit = qt_dense_layer(qt_state)
                print('qt_logit shape: ', qt_logit.get_shape())
                self.qt_loss = focal_loss(self.qtype, qt_logit)
                #self.qt_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.qtype, logits = qt_logit))
                batch_qword = tf.tile(tf.expand_dims(self.qword, 0), [opts.batch_size, 1])
                print('batch_qword shape: ', batch_qword.get_shape()) # B, 7
                top1_qword_indx = tf.nn.top_k(tf.nn.softmax(qt_logit), 1)[1]
                self.pre_qword = tf.batch_gather(batch_qword, top1_qword_indx) 
                print('pre_qword shape: ', self.pre_qword.get_shape())
                
            
            with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):               
                
                # main = tf.strided_slice(self.response, [0, 0], [opts.batch_size, -1], [1, 1])
                # self.decoder_input = tf.concat([tf.fill([opts.batch_size, 1], opts.go_index), main], 1)
                main = tf.strided_slice(self.response, [0, 0], [opts.batch_size, -1], [1, 1])
                self.decoder_input = tf.concat([self.pre_qword, main], 1) # Start decoder with the question word       

                go_index = tf.squeeze(self.pre_qword, 1)
                print('go_index shape: ', go_index.get_shape())
            
                in_decode_con = rnn_output
                in_decode_recall = recall_topics_embed
                in_decode_transit = transit_topics_embed
                
                initial_state = rnn_state
                # Define the decoder cell and the output layer.
                dec_gru_cell = tf.nn.rnn_cell.GRUCell(opts.num_hidden_units)
                output_layer = tf.layers.Dense(units = opts.vocabulary_size,
                                               kernel_initializer = tf.truncated_normal_initializer(stddev = 0.1))
                # output_layer = taware_layer.JointDenseLayer(opts.vocabulary_size, opts.vocabulary_size, name="output_projection")
                
                
                if opts.beam_width > 0 and not opts.training_mode:
                    in_decode_con = tf.contrib.seq2seq.tile_batch(in_decode_con, multiplier=opts.beam_width)
                    in_decode_recall = tf.contrib.seq2seq.tile_batch(in_decode_recall, multiplier=opts.beam_width)
                    in_decode_transit = tf.contrib.seq2seq.tile_batch(in_decode_transit, multiplier=opts.beam_width)
                    initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=opts.beam_width)
                    #in_decode_topic_len = tf.contrib.seq2seq.tile_batch(in_decode_topic_len, multiplier=opts.beam_width)
                    #in_decode_con_len = tf.contrib.seq2seq.tile_batch(in_decode_con_len, multiplier=opts.beam_width)
                
                con_attention = tf.contrib.seq2seq.LuongAttention(num_units = opts.num_hidden_units, memory = in_decode_con)
                recall_attention = tf.contrib.seq2seq.LuongAttention(num_units = opts.num_hidden_units, memory = in_decode_recall)
                transit_attention = tf.contrib.seq2seq.LuongAttention(num_units = opts.num_hidden_units, memory = in_decode_transit)
                
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(dec_gru_cell,attention_mechanism=(con_attention, recall_attention,transit_attention),\
                    attention_layer_size=(opts.num_hidden_units, opts.num_hidden_units,opts.num_hidden_units), alignment_history= True, name="joint_attention")                
                
                
                
                # Perform training decoding.                
                if opts.training_mode:                    
                    training_helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs = tf.nn.embedding_lookup(embeddings,self.decoder_input),
                        sequence_length = self.response_len)
                    
                    # training_decoder = taware_decoder.ConservativeBasicDecoder(decoder_cell,training_helper,\
                    #     decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state), output_layer)

                    training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = decoder_cell,
                    helper = training_helper,
                    initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),
                    output_layer = output_layer)

                    # Dynamic decoding
                    
                    training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder = training_decoder,
                        impute_finished = True,
                        maximum_iterations = 2*tf.reduce_max(self.response_len),
                        swap_memory=True)
                    
                    self.training_logits = training_decoder_output.rnn_output
                    print('==================training_logits:', self.training_logits.get_shape())

                    # predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = embeddings,\
                    #         start_tokens = tf.tile(tf.constant([opts.go_index], dtype=tf.int32), [opts.batch_size]),end_token = opts.eos_index)
                    predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = embeddings,\
                            start_tokens = go_index, end_token = opts.eos_index)
                

                    # predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,\
                    #     initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)
                    predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,\
                        initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)
                    
                    predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = predicting_decoder,\
                        impute_finished = False, maximum_iterations = 2*tf.reduce_max(self.response_len))
                    
                    self.predicting_ids = predicting_decoder_output.sample_id
                
                    # Compute loss function.
                    current_ts = tf.to_int32(tf.minimum(tf.shape(self.response)[-1], tf.shape(self.training_logits)[1]))
                    # # 对 output_out 进行截取
                    new_response = tf.slice(self.response, begin=[0, 0], size=[-1, current_ts])

                    masks = tf.sequence_mask(self.response_len, dtype=tf.float32)
                    
                    self.dec_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits, targets = new_response, weights = masks)
                    
                    self.params = tf.trainable_variables()
                    
                    self.recall_lambda = tf.Variable(float(opts.recall_lambda), trainable=True, dtype=tf.float32)
                    self.transit_lambda = tf.Variable(float(opts.transit_lambda), trainable=True, dtype=tf.float32)
                    self.qtype_lambda = tf.Variable(float(opts.qtype_lambda), trainable=True, dtype=tf.float32)
                    self.recall_l_decay_op = self.recall_lambda.assign(self.recall_lambda * opts.lambda_decay)
                    self.transit_l_decay_op = self.transit_lambda.assign(self.transit_lambda * opts.lambda_decay)
                    self.qtype_l_decay_op = self.qtype_lambda.assign(self.qtype_lambda * opts.lambda_decay)
                    
                    self.lr = tf.Variable(float(opts.learning_rate), trainable=True, dtype=tf.float32)

                    self.lr_decay_op = self.lr.assign(self.lr * opts.lr_decay)
                    
                    self.loss = self.dec_loss + self.recall_lambda * self.recall_loss + self.transit_lambda * self.transit_loss + self.qtype_lambda * self.qt_loss
                    
                    
                    self.global_step = tf.Variable(0, trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate =self.lr)
                    
                    gradients = tf.gradients(self.loss, self.params)
                    clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,opts.max_grad_norm)

                    self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)


                   

                else:
                    if opts.beam_width > 0:
                        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, 
                                                                            embedding=embeddings,
                                                                            start_tokens=tf.tile(tf.constant([opts.go_index], dtype=tf.int32),[opts.batch_size]), 
                                                                            end_token=opts.eos_index,
                                                                            initial_state=decoder_cell.zero_state(opts.batch_size*opts.beam_width, tf.float32).clone(cell_state=initial_state),
                                                                            beam_width=opts.beam_width,
                                                                            output_layer=output_layer)
                    
                    else:
                        # predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = embeddings,\
                        #     start_tokens = tf.tile(tf.constant([opts.go_index], dtype=tf.int32), [opts.batch_size]),end_token = opts.eos_index)
                        
                        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = embeddings,\
                            start_tokens = go_index, end_token = opts.eos_index)
                    

                        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,\
                            initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)
                        #predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,helper = predicting_helper,
                        # initial_state = decoder_cell.zero_state(opts.batch_size, tf.float32).clone(cell_state=initial_state),output_layer = output_layer)

                    
                    predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = predicting_decoder,\
                        impute_finished = False, maximum_iterations = 2*tf.reduce_max(self.response_len))
                    #self.training_logits = training_decoder_output.rnn_output
                    if opts.beam_width > 0:
                        self.predicting_ids = predicting_decoder_output.predicted_ids[:,:,0]
                    else:
                        self.predicting_ids = predicting_decoder_output.sample_id

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=3)
                

                
            
    def init_tf_vars(self):
        self.session.run(self.init)
        print('TensorFlow variables initialized.')
    
    def bleu(self, ref, pre):
        ref = ref.split(' ')
        pre = pre.split(' ')
        bleu_1 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(1, 0, 0, 0),nltk.translate.bleu_score.SmoothingFunction().method1)
        bleu_2 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(0.5,0.5),nltk.translate.bleu_score.SmoothingFunction().method1)
        bleu_3 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(0.333,0.333,0.333),nltk.translate.bleu_score.SmoothingFunction().method1)
        bleu_4 = nltk.translate.bleu_score.sentence_bleu([ref],pre,(0.25, 0.25, 0.25, 0.25),nltk.translate.bleu_score.SmoothingFunction().method1)
        return bleu_1, bleu_2, bleu_3, bleu_4

    def distinct_1(self, lines):
        '''Computes the number of distinct words divided by the total number of words.

        Input:
        lines: List String.
        
        '''
        words = ' '.join(lines).split(' ')
        #if '<EOS>' in words:
        #    words.remove('<EOS>')
        num_distinct_words = len(set(words))
        return float(num_distinct_words) / len(words)


    def distinct_2(self, lines):
        '''Computes the number of distinct bigrams divided by the total number of words.

        Input:
        lines: List of strings.
        '''
        all_bigrams = []
        num_words = 0

        for line in lines:
            line_list = line.split(' ')
            #if '<EOS>' in line_list:
            #    line_list.remove('<EOS>')            
            num_words += len(line_list)
            bigrams = zip(line_list, line_list[1:])
            all_bigrams.extend(list(bigrams))

        return len(set(all_bigrams)) / float(num_words)
    


    # ========== Our own embedding-based metric ========== #
    def cal_vector_extrema(self,x, y, dic):
        # x and y are the list of the words
        # dic is the gensim model which holds 300 the google news word2ved model
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    #vectors.append(dic[w.lower()])
                    vectors.append(dic[w])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)
        vec_x = np.max(x, axis=0)
        vec_y = np.max(y, axis=0)
        assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
        zero_list = np.zeros(len(vec_x))
        if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
            return float(1) if vec_x.all() == vec_y.all() else float(0)
        res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return cos


    def cal_embedding_average(self, x, y, dic):
        # x and y are the list of the words
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    #vectors.append(dic[w.lower()])
                    vectors.append(dic[w])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)
        
        vec_x = np.array([0 for _ in range(len(x[0]))])
        for x_v in x:
            x_v = np.array(x_v)
            vec_x = np.add(x_v, vec_x)
        vec_x = vec_x / math.sqrt(sum(np.square(vec_x)))
        
        vec_y = np.array([0 for _ in range(len(y[0]))])
        #print(len(vec_y))
        for y_v in y:
            y_v = np.array(y_v)
            vec_y = np.add(y_v, vec_y)
        vec_y = vec_y / math.sqrt(sum(np.square(vec_y)))
        
        assert len(vec_x) == len(vec_y), "len(vec_x) != len(vec_y)"
        
        zero_list = np.array([0 for _ in range(len(vec_x))])
        if vec_x.all() == zero_list.all() or vec_y.all() == zero_list.all():
            return float(1) if vec_x.all() == vec_y.all() else float(0)
        
        vec_x = np.mat(vec_x)
        vec_y = np.mat(vec_y)
        num = float(vec_x * vec_y.T)
        denom = np.linalg.norm(vec_x) * np.linalg.norm(vec_y)
        cos = num / denom
        
        # res = np.array([[vec_x[i] * vec_y[i], vec_x[i] * vec_x[i], vec_y[i] * vec_y[i]] for i in range(len(vec_x))])
        # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        
        return cos


    def cal_greedy_matching(self, x, y, dic):
        # x and y are the list of words
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    #vectors.append(dic[w.lower()])
                    vectors.append(dic[w])
            if not vectors:
                #vectors.append(np.random.randn(300))
                vectors.append(np.zeros(300))
            return np.stack(vectors)
        x = vecterize(x)
        y = vecterize(y)
        
        len_x = len(x)
        len_y = len(y)
        
        cosine = []
        sum_x = 0 

        for x_v in x:
            for y_v in y:
                assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
                zero_list = np.zeros(len(x_v))

                if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                    if x_v.all() == y_v.all():
                        cos = float(1)
                    else:
                        cos = float(0)
                else:
                    # method 1
                    res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

                cosine.append(cos)
            if cosine:
                sum_x += max(cosine)
                cosine = []

        sum_x = sum_x / len_x
        cosine = []

        sum_y = 0

        for y_v in y:

            for x_v in x:
                assert len(x_v) == len(y_v), "len(x_v) != len(y_v)"
                zero_list = np.zeros(len(y_v))

                if x_v.all() == zero_list.all() or y_v.all() == zero_list.all():
                    if (x_v == y_v).all():
                        cos = float(1)
                    else:
                        cos = float(0)
                else:
                    # method 1
                    res = np.array([[x_v[i] * y_v[i], x_v[i] * x_v[i], y_v[i] * y_v[i]] for i in range(len(x_v))])
                    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

                cosine.append(cos)

            if cosine:
                sum_y += max(cosine)
                cosine = []

        sum_y = sum_y / len_y
        score = (sum_x + sum_y) / 2
        return score


    def cal_greedy_matching_matrix(self, x, y, dic):
        # x and y are the list of words
        def vecterize(p):
            vectors = []
            for w in p:
                if w in dic:
                    vectors.append(dic[w])
            if not vectors:
                vectors.append(np.random.randn(300))
            return np.stack(vectors)
        x = vecterize(x)     # [x, 300]
        y = vecterize(y)     # [y, 300]
        
        len_x = len(x)
        len_y = len(y)
        
        matrix = np.dot(x, y.T)    # [x, y]
        matrix = matrix / np.linalg.norm(x, axis=1, keepdims=True)    # [x, 1]
        matrix = matrix / np.linalg.norm(y, axis=1).reshape(1, -1)    # [1, y]
        
        x_matrix_max = np.mean(np.max(matrix, axis=1))    # [x]
        y_matrix_max = np.mean(np.max(matrix, axis=0))    # [y]
        
        return (x_matrix_max + y_matrix_max) / 2


    def cal_embedding_metric(self, refs, pres, w2v):    
        if len(refs) != len(pres):
            print('Shape Error!')
    
        greedys = []
        averages = []
        extremas = []
        for i in range(len(refs)):
            x = refs[i].strip().split(' ')
            y = pres[i].strip().split(' ')
            greedys.append(self.cal_greedy_matching_matrix(x, y, w2v))
            averages.append(self.cal_embedding_average(x, y, w2v))
            extremas.append(self.cal_vector_extrema(x, y, w2v))
        
        
        return np.mean(np.asarray(averages)), np.mean(np.asarray(extremas)), np.mean(np.asarray(greedys))
    

    def eval(self, pre_lines,tar_lines):
        opts = self.options
        #w2v = 
        # BLEU
        bleu_1 = []
        bleu_2 = []
        bleu_3 = []
        bleu_4 = []
        for i in range(len(pre_lines)):            
            one, two, three, four = self.bleu(tar_lines[i],pre_lines[i])
            bleu_1.append(one)
            bleu_2.append(two)
            bleu_3.append(three)
            bleu_4.append(four)
        avg_bleu_1 = sum(bleu_1)/len(bleu_1)
        avg_bleu_2 = sum(bleu_2)/len(bleu_2)
        avg_bleu_3 = sum(bleu_3)/len(bleu_3)
        avg_bleu_4 = sum(bleu_4)/len(bleu_4)

        # Distinct-1
        dis = Distinct()
        dis_1 = dis.sentence_score(pre_lines, 1)
        # Distinct-2
        dis_2 = dis.sentence_score(pre_lines, 2)
        #Embedding
        #average, extrema, greedy = self.cal_embedding_metric(tar_lines, pre_lines, opts.embedding_file)
        average, extrema, greedy = 0,0,0
        return avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2


    

    def dynamic_train(self, enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, transit_topic, transit_topic_lens, cr_label, tr_label, q_type, q_word):
        print('Start to train the model with the "Dynamic-training strategy"...')
        opts = self.options
        num_examples = enc_x.shape[0]
        num_batches = num_examples // opts.batch_size
        valid_time = 0
        epoch_loss = {'total': [], 'dec':[], 'recall':[], 'transit':[], 'qtype':[]}
        epoch_lambda = {'recall':[], 'transit':[], 'qtype':[]}
        pre_total_losses = [1e18]*5
        pre_recall_losses = [1e8]*2
        pre_transit_losses = [1e8]*2
        pre_qtype_losses = [1e8]*2

        for epoch in range(opts.num_epochs):
            e_loss = 0
            e_dec_loss = 0
            e_recall_loss = 0
            e_transit_loss = 0
            e_q_loss = 0
            e_t_lambda, e_r_lambda, e_q_lambda = 0, 0, 0
            perm_indices = np.random.permutation(range(num_examples))
        
            print('================', 'Epoch ', epoch+1, '================')
                    
            for batch in range(num_batches):
                #if fail_time > patient:
                #    break
                s = batch * opts.batch_size
                t = s + opts.batch_size
                batch_indices = perm_indices[s:t]
                # B turn S--turn,S,B
                batch_enc_x = np.transpose(enc_x[batch_indices,:,:], [1, 2, 0])
                batch_context_lens = context_lens[batch_indices]
                batch_enc_x_lens = np.transpose(enc_x_lens[batch_indices,:])                
                batch_dec_y = dec_y[batch_indices,:]
                batch_dec_y_lens = dec_y_lens[batch_indices]
                batch_x_topic = x_topic[batch_indices,:]
                batch_x_topic_lens = x_topic_lens[batch_indices]
                batch_transit_topic = transit_topic[batch_indices,:]
                batch_transit_topic_lens = transit_topic_lens[batch_indices]
                batch_cr_label = cr_label[batch_indices,:]
                batch_tr_label = tr_label[batch_indices,:]
                batch_q_type = q_type[batch_indices,:]
                batch_q_word = q_word
                
                feed_dict = {self.context: batch_enc_x,
                             self.turn_num: batch_context_lens,
                             self.context_lens: batch_enc_x_lens,
                             self.context_topic: batch_x_topic,
                             self.con_topic_len: batch_x_topic_lens,
                             self.response: batch_dec_y,
                             self.response_len: batch_dec_y_lens,
                             self.transit_topic: batch_transit_topic,
                             self.transit_topic_len: batch_transit_topic_lens,
                             self.cr_label: batch_cr_label,
                             self.tr_label: batch_tr_label,
                             self.qtype: batch_q_type,
                             self.qword: batch_q_word}

                _, loss_total, loss_dec, loss_recall, loss_transit, loss_q, l_r, r_lambda, t_lambda, q_lambda = self.session.run([self.train_op, self.loss, self.dec_loss, self.recall_loss,\
                   self.transit_loss,self.qt_loss,self.lr,self.recall_lambda,self.transit_lambda,self.qtype_lambda], feed_dict = feed_dict)      
                e_loss += loss_total
                e_dec_loss += loss_dec
                e_recall_loss += loss_recall
                e_transit_loss += loss_transit
                e_q_loss += loss_q
                e_r_lambda += r_lambda
                e_t_lambda += t_lambda
                e_q_lambda += q_lambda

                if batch%100==0:
                    if loss_total > max(pre_total_losses):
                        self.session.run(self.lr_decay_op)
                    pre_total_losses = pre_total_losses[1:]+[loss_total]
                    print('total_loss: ', pre_total_losses)
                    
                    
                    # print loss and predicted sentences
                    print_predicted, pre_recall, pre_transit, pre_qword = self.session.run([self.predicting_ids,\
                     self.select_context_topic, self.select_transit_topic, self.pre_qword], feed_dict= feed_dict)
                    print('------------------------------------------------')
                    print('Epoch {:03d} batch {:04d}/{:04d} Learning Rate {}'.format(epoch + 1, batch + 1,num_batches, l_r), flush = True)
                    
                    print('total_loss: {}, dec_loss: {}, recall_loss: {}, transit_loss: {}, qt_loss: {}'.format(loss_total, \
                    loss_dec, loss_recall, loss_transit, loss_q), flush = True)
                    print('recall_lambda: {}, transit_lambda: {}, qtype_labda: {}'.format(r_lambda, \
                    t_lambda, q_lambda), flush = True)
                     
                    print('----------------  Samples  ----------------')
                    print('Last CONTEXT:',' '.join([opts.reverse_vocabulary[n] for n in batch_enc_x[batch_context_lens[0]-1,:,0] if n not in [0,1,2]]))                    
                    #print('Transit Topic: ', ' '.join([opts.reverse_vocabulary[n] for n in batch_transit_topic[0] if n not in [0,1,2]]))
                    print('Recall-Topic: ', ' '.join([opts.reverse_vocabulary[n] for n in pre_recall[0] if n not in [0,1,2]]))
                    print('Transit-Topic: ', ' '.join([opts.reverse_vocabulary[n] for n in pre_transit[0] if n not in [0,1,2]]))
                    print('Predict Qword: ', opts.reverse_vocabulary[pre_qword[0][0]])
                    
                    flag = True
                    pre = []
                    for n in print_predicted[0]: 
                        if flag:
                            if n==2:
                                flag = False
                            else:
                                if n not in [0, 1]:
                                    pre.append(opts.reverse_vocabulary[n])
                        else:
                            pass      
                    
                    
                    pre_sen = ' '.join(pre)
                    ref_sen = ' '.join([opts.reverse_vocabulary[n] for n in batch_dec_y[0] if n not in [0,1,2]])
                    print('----------------')
                    print('REF:',ref_sen)
                    print('PRE:',pre_sen)
                    print()
                    
                    
                if batch%300==0 and batch != 0:
                    # validate and save model
                    print('********************************')
                    valid_time += 1
                    print('The {:03d} Validation Time. Epoch {:03d} Batch {:03d}'.format(valid_time, epoch+1, batch+1))                    
                    pre_lines, ref_lines = self.valid(opts.dev_data_path, valid_time, q_word)
                    
                    
                    
                    with open(opts.save_model_path + '-'+str(valid_time) + '-result.txt', 'w', encoding ='utf-8') as f:
                        for i in range(len(ref_lines)):
                            f.write('ref: ' + ref_lines[i])
                            f.write('\n')
                            f.write('pre: ' + pre_lines[i])
                            f.write('\n')                            
                            f.write('\n')
                        f.close()                   
                    
                    print('********************************')
                    print('')

                    
            print('********************************')           
            print('Epoch {:03d} Avg. total_loss:{}, dec_loss:{}, recall_loss:{}, transit_loss:{}, qt_loss: {}'.format(epoch + 1, e_loss/num_batches,\
                e_dec_loss/num_batches,e_recall_loss/num_batches, e_transit_loss/num_batches, e_q_loss/num_batches), flush = True)
            
            self.saver.save(self.session, opts.save_model_path + '/model_epoch_%02d' % (epoch+1))
            print('Model saved!')
            if e_recall_loss/num_batches > np.mean(np.array(pre_recall_losses)):
                self.session.run(self.recall_l_decay_op)
            pre_recall_losses = pre_recall_losses[1:] + [e_recall_loss/num_batches]
            print('recall_loss: ', pre_recall_losses)
            if e_transit_loss/num_batches > np.mean(np.array(pre_transit_losses)):
                self.session.run(self.transit_l_decay_op)
            pre_transit_losses = pre_transit_losses[1:] + [e_transit_loss/num_batches]
            print('transit_loss: ', pre_transit_losses)
            if e_q_loss/num_batches > np.mean(np.array(pre_qtype_losses)):
                self.session.run(self.qtype_l_decay_op)
            pre_qtype_losses = pre_qtype_losses[1:] + [e_q_loss/num_batches]
            print('qtype_loss: ', pre_qtype_losses)
            
            print('********************************')
            print('')
            epoch_loss['total'].append(e_loss/num_batches)
            epoch_loss['dec'].append(e_dec_loss/num_batches)
            epoch_loss['recall'].append(e_recall_loss/num_batches)
            epoch_loss['transit'].append(e_transit_loss/num_batches)
            epoch_loss['qtype'].append(e_q_loss/num_batches)
            epoch_lambda['recall'].append(e_r_lambda/num_batches)
            epoch_lambda['transit'].append(e_t_lambda/num_batches)
            epoch_lambda['qtype'].append(e_q_lambda/num_batches)

            w2f(opts.save_model_path+'/epoch_recall_lambda.txt', epoch_lambda['recall'])
            w2f(opts.save_model_path+'/epoch_transit_lambda.txt', epoch_lambda['transit'])
            w2f(opts.save_model_path+'/epoch_qtype_lambda.txt', epoch_lambda['qtype'])
            w2f(opts.save_model_path+'/epoch_total_loss.txt', epoch_loss['total'])
            w2f(opts.save_model_path+'/epoch_dec_loss.txt', epoch_loss['dec'])
            w2f(opts.save_model_path+'/epoch_recall_loss.txt', epoch_loss['recall'])
            w2f(opts.save_model_path+'/epoch_transit_loss.txt', epoch_loss['transit'])
            w2f(opts.save_model_path+'/epoch_qtype_loss.txt', epoch_loss['qtype'])
        
        
        

        print('')



    def save(self, save_path):
        print('Saving the trained model...')
        self.saver.save(self.session, save_path)

    def restore(self, restore_path):
        print('Restoring from a pre-trained model...')        
        self.saver.restore(self.session, tf.train.latest_checkpoint(restore_path))
    
    
    
    def valid(self, data_path, valid_time, q_word):                          
        print('Start Validate the model......')
        enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, transit_topic, transit_topic_lens, cr_label, tr_label, q_type\
             = read_data(data_path)    
        opts = self.options
        num_examples = enc_x.shape[0]
        num_batches = num_examples // opts.batch_size        
        
        total_loss, dec_loss, recall_loss, transit_loss, q_loss = 0, 0, 0, 0, 0
        ref_lines = []
        pre_lines = []  
        
        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            batch_enc_x = np.transpose(enc_x[s:t,:,:], [1, 2, 0])
            batch_context_lens = context_lens[s:t]
            batch_enc_x_lens = np.transpose(enc_x_lens[s:t,:])                
            batch_dec_y = dec_y[s:t,:]
            batch_dec_y_lens = dec_y_lens[s:t]
            batch_x_topic = x_topic[s:t,:]
            batch_x_topic_lens = x_topic_lens[s:t]
            batch_transit_topic = transit_topic[s:t,:]
            batch_transit_topic_lens = transit_topic_lens[s:t]
            batch_cr_label = cr_label[s:t,:]
            batch_tr_label = tr_label[s:t,:]
            batch_q_type = q_type[s:t,:]
            batch_q_word = q_word

            feed_dict = {self.context: batch_enc_x,
                             self.turn_num: batch_context_lens,
                             self.context_lens: batch_enc_x_lens,
                             self.context_topic: batch_x_topic,
                             self.con_topic_len: batch_x_topic_lens,
                             self.response: batch_dec_y,
                             self.response_len: batch_dec_y_lens,
                             self.transit_topic: batch_transit_topic,
                             self.transit_topic_len: batch_transit_topic_lens,
                             self.cr_label: batch_cr_label,
                             self.tr_label: batch_tr_label,
                             self.qtype: batch_q_type,
                             self.qword: batch_q_word}

            loss_total, loss_dec, loss_recall, loss_transit, loss_q, predicted = self.session.run([self.loss,self.dec_loss,self.recall_loss,self.transit_loss,\
                 self.qt_loss, self.predicting_ids], feed_dict = feed_dict)
            total_loss += loss_total
            dec_loss += loss_dec
            recall_loss += loss_recall
            transit_loss += loss_transit
            q_loss += loss_q            

            for i in range(opts.batch_size):
                
                ref = ' '.join([opts.reverse_vocabulary[n] for n in batch_dec_y[i] if n not in[0,1,2]])
                flag = True
                pre = []
                for n in predicted[i]: 
                    if flag:
                        if n==2:
                            flag = False
                        else:
                            if n not in [0,1]:
                                pre.append(opts.reverse_vocabulary[n])
                    else:
                        pass      

                #pre = ' '.join([opts.reverse_vocabulary[n] for n in predicted[i] if n not in[0,1,2]])
                pre = ' '.join(pre)
                ref_lines.append(ref)
                pre_lines.append(pre)
                
        avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2 = self.eval(pre_lines,ref_lines)
        print('total_loss: {}, dec_loss: {}, recall_loss: {}, transit_loss:{}, q_loss:{}'.format(total_loss/num_batches, dec_loss/num_batches,\
            recall_loss/num_batches, transit_loss/num_batches, q_loss/num_batches))
        print('BLEU: {}, {}, {}, {}'.format(avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4))
        # print('Valid Embedding: {}, {}, {}'.format(average, extrema, greedy))        
        print('DISTINCT: {}, {}'.format(dis_1, dis_2)) 
        return pre_lines, ref_lines

    def predict(self, data_path, q_word): 
        print('Start Test the model......')
        enc_x, context_lens, enc_x_lens, dec_y, dec_y_lens, x_topic, x_topic_lens, transit_topic, transit_topic_lens, cr_label, tr_label, q_type\
             = read_data(data_path)    
        opts = self.options
        num_examples = enc_x.shape[0]
        num_batches = num_examples // opts.batch_size        
        
        dec_loss = 0
        ref_lines = []
        pre_lines = []  
        test_samples = []

        for batch in range(num_batches):
            s = batch * opts.batch_size
            t = s + opts.batch_size
            batch_enc_x = np.transpose(enc_x[s:t,:,:], [1, 2, 0])
            batch_context_lens = context_lens[s:t]
            batch_enc_x_lens = np.transpose(enc_x_lens[s:t,:])                
            batch_dec_y = dec_y[s:t,:]
            batch_dec_y_lens = dec_y_lens[s:t]
            batch_x_topic = x_topic[s:t,:]
            batch_x_topic_lens = x_topic_lens[s:t]
            batch_transit_topic = transit_topic[s:t,:]
            batch_transit_topic_lens = transit_topic_lens[s:t]
            batch_cr_label = cr_label[s:t,:]
            batch_tr_label = tr_label[s:t,:]
            batch_q_type = q_type[s:t,:]
            batch_q_word = q_word

            feed_dict = {self.context: batch_enc_x,
                             self.turn_num: batch_context_lens,
                             self.context_lens: batch_enc_x_lens,
                             self.context_topic: batch_x_topic,
                             self.con_topic_len: batch_x_topic_lens,
                             self.response: batch_dec_y,
                             self.response_len: batch_dec_y_lens,
                             self.transit_topic: batch_transit_topic,
                             self.transit_topic_len: batch_transit_topic_lens,
                             self.cr_label: batch_cr_label,
                             self.tr_label: batch_tr_label,
                             self.qtype: batch_q_type,
                             self.qword: batch_q_word}

            predicted, pre_recall, pre_transit, pre_qword = self.session.run([self.predicting_ids, self.select_context_topic,\
            self.select_transit_topic, self.pre_qword], feed_dict = feed_dict)
            
            
            for i in range(opts.batch_size):
                con = []
                for line in batch_enc_x[:,:,i]:                    
                    str_line = ' '.join([opts.reverse_vocabulary[n] for n in line if n not in[0,1,2]])
                    if len(str_line)>1 or str_line not in ['',' ']:
                        con.append(str_line) 
                context = ' || '.join(con)
                #transit_topic_utt = ' '.join([opts.reverse_vocabulary[n] for n in batch_transit_topic[i] if n not in[0,1,2]])
                pre_recall_utt = ' '.join([opts.reverse_vocabulary[n] for n in pre_recall[i] if n not in[0,1,2]])
                pre_transit_utt = ' '.join([opts.reverse_vocabulary[n] for n in pre_transit[i] if n not in[0,1,2]])
                pre_qword_utt = opts.reverse_vocabulary[pre_qword[0][0]]
                ref = ' '.join([opts.reverse_vocabulary[n] for n in batch_dec_y[i] if n not in[0,1,2]])
                pre = []
                for n in predicted[i]:
                    if n in [0,1,2]:
                        break
                    else:
                        pre.append(opts.reverse_vocabulary[n])
                pre = ' '.join(pre)
                #pre = ' '.join([opts.reverse_vocabulary[n] for n in predicted[i] if n not in[0,1,2]])
                ref_lines.append(ref)
                pre_lines.append(pre)
                

                sample = dict()
                sample['Context']=context
                sample['Recall']=pre_recall_utt
                sample['Transit'] = pre_transit_utt
                sample['Qword'] = pre_qword_utt
                sample['Target'] = ref
                sample['Predict'] = pre
                
                test_samples.append(sample)
        avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, dis_1, dis_2= self.eval(pre_lines,ref_lines)
        print('TEST BLEU: {}, {}, {}, {}'.format(avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4))
        print('TEST DISTINCT: {}, {}'.format(dis_1, dis_2))   
        
            
        
        return test_samples


