#!/usr/bin/python
"""
Toy example of attention layer use

Train RNN (GRU) on IMDB dataset (binary classification)
Learning and hyper-parameters were not tuned; script serves as an example 
"""
from __future__ import print_function, division

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicRNNCell as GRUCell
#from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from keras.preprocessing.text import Tokenizer 
from attention import attention
from utils import get_vocabulary_size, batch_generator
import itertools  
import pickle
import heapq
#%%
def train_RNN(SEQUENCE_LENGTH, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, KEEP_PROB, BATCH_SIZE, NUM_EPOCHS, DELTA, LEARNING_RATE):    

    # Load preprocessed data 
    with open ('raw_sums', 'rb') as fp:
        raw_sums = pickle.load(fp)
    
    with open ('raw_texts', 'rb') as fp:
        raw_texts = pickle.load(fp)
    
    with open ('vocab_limit', 'rb') as fp:
        vocab_limit = pickle.load(fp)
  
    
    # Embedd vocabulary(important for our int2word later to have only one embedding)
    embd_vocab = []             
    max_words = len(vocab_limit)
    tokenizer = Tokenizer(num_words=max_words)
    # This builds the word index
    tokenizer.fit_on_texts(vocab_limit)
    # This turns strings into lists of integer indices.
    embd_vocab = tokenizer.texts_to_sequences(vocab_limit)
    embd_vocab = list(itertools.chain(*embd_vocab))
    #embd_vocab.append(encoded_docs)
    print('TRAIN_ARNN: vocab embedded.')
    print('Länge EMBD_VOCAB: ' + str(len(embd_vocab)))
    
    # Saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    keys = vocab_limit[:]
    values = embd_vocab[:]
    dictionary = dict(zip(keys, values))
    
    
    def word2int(texts):
        embd_texts = []
        for each_text in texts: 
            embd_text = []
            for word in each_text: 
                for key in dictionary.keys():
                    if key == word: 
                        embd_text.append(dictionary.get(key))
            embd_texts.append(embd_text)
        return embd_texts
        print('TRAIN_ARNN: ' + str(texts) + ' embedded.')
        
 
    embd_texts = word2int(raw_texts)
    embd_sums = word2int(raw_sums)
    
    print(embd_texts[0])
    #%% ADDED BY MKLOENNE from summarization model -- Creating train and test sets
    
    train_len = int((.7)*len(embd_sums))
    
    train_texts = embd_texts[0:train_len]
    train_summaries = embd_sums[0:train_len]
    
    val_len = int((.15)*len(embd_sums))
    '''
    val_texts = embd_texts[train_len:train_len+val_len]
    val_summaries = embd_sums[train_len:train_len+val_len]
    '''
    test_texts = embd_texts[train_len+val_len:len(embd_sums)]
    test_summaries = embd_sums[train_len+val_len:len(embd_sums)]
    
    
    # Load the dataset 
    (X_train, y_train), (X_test, y_test) = (train_texts, train_summaries), (test_texts, test_summaries)

    # Convert text lists into numpy arrays to get shape like in the imdb dataset for train and test data 
    def convert2array(listtoconvert, comparing):
        converted_arr = []
        maxList = max(max(len(x) for x in listtoconvert), max(len(x) for x in comparing))
        pre_array = np.asarray([np.asarray(x) for x in listtoconvert])
        for arr in pre_array:
            arr = np.lib.pad(arr, (0,maxList-len(arr)), 'constant', constant_values=0)
            text_with_zero = np.concatenate([[1],arr])
            text_with_zero = np.concatenate([text_with_zero,[0]])
            converted_arr.append(text_with_zero)
        converted_arr = np.asarray(converted_arr)
        return converted_arr
    
    max_y = max(max(len(x) for x in train_summaries), max(len(x) for x in test_summaries))+2
    X_test = convert2array(test_texts, train_texts)
    X_train = convert2array(train_texts, test_texts)
    y_test = convert2array(test_summaries, train_summaries)
    y_train = convert2array(train_summaries, test_summaries)
    
    #%%

    vocabulary_size = get_vocabulary_size(X_train) 

    # Different placeholders
    batch_ph = tf.placeholder(tf.int32, [None, None], name="X")
    target_ph = tf.placeholder(tf.float32, [None, None], name="Y") #adapted for y 
    seq_len_ph = tf.placeholder(tf.int32, [None], name="SEQ_LEN")
    keep_prob_ph = tf.placeholder(tf.float32, name="KEEP_PROB")
    tf.summary.scalar('dropout_keep_probability', KEEP_PROB)

    with tf.name_scope("Embedding-Layer"):
        # Embedding layer
        embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True, name="embedding")
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)
     
    with tf.name_scope("RNN"):    
        # (Bi-)RNN layer(-s)
        rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                                inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
        #, _ = rnn(GRUCell(HIDDEN_SIZE), inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
        
    print('EMBEDDED:' + str(batch_embedded[0]))
    
    with tf.name_scope("Attention-Layer"):
        # Attention layer
        attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    
    with tf.name_scope("Dropout"):
        # Dropout
        drop = tf.nn.dropout(attention_output, keep_prob_ph)
   
    # Fully connected layer
    W = tf.Variable(tf.truncated_normal(shape=[HIDDEN_SIZE * 2, max_y], stddev=0.1), name="weights")  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[max_y]), name="bias") 
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    tf.summary.histogram("y_hat", y_hat)
    
    with tf.name_scope("Costs"):
        # Cross-entropy loss and optimizer initialization
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph)) 
    tf.summary.scalar('Costs', loss)
    with tf.name_scope("Optimize"):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    with tf.name_scope("Accuracy"):
        # Accuracy metric
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))
    tf.summary.scalar('Accuracy', accuracy)

    # Batch generators
    train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
    test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

#%%

    saver = tf.train.Saver()
    # Train session 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./logs/nn_logs' + '/train')
#                                              sess.graph)
        train_writer.add_graph(sess.graph)
        test_writer = tf.summary.FileWriter('./logs/nn_logs' + '/test')
        test_writer.add_graph(sess.graph)
        merged = tf.summary.merge_all()
        
        print("TRAIN_ARNN: Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = next(train_batch_generator)
                maxValue_y = max(map(lambda x: x[len(y_batch)], y_batch))
                j=0
                y_batch_normed=[]
                for each_y_batch in y_batch:
                    norm_y = [float(i)/maxValue_y for i in each_y_batch]
                    y_batch_normed.append(norm_y)
                    j += 1                
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                summary, loss_tr, acc, _ = sess.run([merged, loss, accuracy, optimizer],
                                           feed_dict={batch_ph: x_batch,
                                                      target_ph: y_batch_normed,
                                                      seq_len_ph: seq_len,
                                                      keep_prob_ph: KEEP_PROB}) 

                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                train_writer.add_summary(summary, epoch * num_batches + b)    
            if accuracy_train != 0:
                accuracy_train /= num_batches
#            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
#                loss_train, loss_test, accuracy_train, accuracy_test
#            ))
            

            # Testing
            print(X_test.shape)
            num_batches = X_test.shape[0] // BATCH_SIZE
            for b in range(num_batches):
                x_batch, y_batch = next(test_batch_generator)
                maxValue_y = max(map(lambda x: x[len(y_batch)], y_batch))
                j=0
                y_batch_normed=[]
                for each_y_batch in y_batch:
                    norm_y = [float(i)/maxValue_y for i in each_y_batch]
                    y_batch_normed.append(norm_y)
                    j += 1       
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                summary, loss_test_batch, acc = sess.run([merged, loss, accuracy],
                                                feed_dict={batch_ph: x_batch,
                                                           target_ph:  y_batch_normed,
                                                           seq_len_ph: seq_len,
                                                           keep_prob_ph: 1.0}) 
                print('Test-loss:' + str(loss_test_batch))
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, epoch * num_batches + b)
            if loss_test and accuracy_test == 0:
                print('loss_test, acc_test is zero!')
                break
            accuracy_test /= num_batches
            loss_test /= num_batches
            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, accuracy_train, accuracy_test
            ))
            

        #saver.save(sess, "H:/AnacondaProjects/tf-rnn-attention/tf-rnn-attention-master/model.ckpt") # "model")
        saver.save(sess, "/home/kloe_mr/AnacondaProjects/tf-rnn-attention/tf-rnn-attention-master/model.ckpt") # "model")
        
        
#%% Restore session 
        with tf.Session() as sess:
            saver.restore(sess, "/home/kloe_mr/AnacondaProjects/tf-rnn-attention/tf-rnn-attention-master/model.ckpt")#"model")
            #saver.restore(sess, "H:/AnacondaProjects/tf-rnn-attention/tf-rnn-attention-master/model.ckpt")#"model")
            
            x_batch_train, y_batch_train = X_train[:len(X_train)], y_train[:len(y_train)]
            seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_train])
            maxValue_y = max(map(lambda x: x[len(y_batch_train)], y_batch_train))
            j=0
            y_batch_normed=[]
            for each_y_batch in y_batch_train:
                norm_y = [float(i)/maxValue_y for i in each_y_batch]
                y_batch_normed.append(norm_y)
                j += 1
            alphas_test = sess.run([alphas], feed_dict={batch_ph: x_batch_train, target_ph: y_batch_normed,
                                                        seq_len_ph: seq_len_test, keep_prob_ph: 1.0})
    
    
    
#%% setting up the generated outputs by choosing the most weighted higlights out of the text for the LSTM input 
    
        alphas_values = alphas_test[:][0]
        #print(alphas_values[0])
        # Save visualization as HTML
        rnn_outs = []
        #with open("visualization.html", "w") as html_file:
        for words, alphas in zip(x_batch_train, alphas_values):
            rnn_out=[]
            Largest_alphas = len(words)//10
            min_value = min(heapq.nlargest(Largest_alphas, alphas, key=None))
            for word, alpha in zip(words, alphas):#_values):# / alphas_values.max()):
                if alpha > min_value: 
                    if word not in rnn_out:
                        rnn_out.append(word)
                if word == 0:
                    break
                #html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))
            rnn_outs.append(rnn_out)
        print('RNN_OUTS: '+ str(len(rnn_outs)))
 
    # Creates list of lists with reembedded words
    def int2word(data):
        lists = []
        for paragraph in data:
            words = []
            for word in paragraph:
                if word == 1:
                    continue
                if word == 0: 
                    break
                words.append(list(dictionary.keys())[list(dictionary.values()).index(word)]) 
            lists.append(words)  
        return lists

    rnn_out = int2word(rnn_outs)
    batch_words = int2word(y_batch_train)
    print(rnn_out[0])
    print(batch_words[0])
    
    
#%%
     # saving
    with open('rnn_out.pickle', 'wb') as handle:
        pickle.dump(rnn_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # saving
    with open('batch_words.pickle', 'wb') as handle:
        pickle.dump(batch_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

     
