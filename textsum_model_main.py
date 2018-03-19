# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 23:35:27 2018

@author: kloe_mr
"""

from prepare_cnn_data import data_preparation
from remove_stopwords_tensorboard_arnn import train_RNN
from sum_model_part2 import train_LSTM

MAXIMUM_DATA_NUM = 500
#data_directory = 'C:/Users/kloe_mr/Documents/data/cnn_stories/cnn/stories/'
data_directory = '/home/kloe_mr/AnacondaProjects/data/cnn_stories/cnn/stories/'  ## for linux workstation 

#HYPERPARAMETERS FOR THE aRNN 
SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100
HIDDEN_SIZE = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 25
#EXPLANATION' FOR BATCHES: https://theneuralperspective.com/2016/10/04/05-recurrent-neural-networks-rnn-part-1-basic-rnn-char-rnn/
NUM_EPOCHS = 3  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
LEARNING_RATE = 0.003
ALPHA_DIVIDER = 15

#HYPERPARAMETERS FOR THE LSTMS
MAX_SUMMARY_LEN = 100
MAX_TEXT_LEN = 1000   
#D is a major hyperparameters. Windows size for local attention will be 2*D+1
D = 5
LSTM_HIDDEN_SIZE = 350#500
LEARNING_RATE = 0.003
K = 5#5
TRAINING_ITERS = 15 

#runs the data preprocessing 
#data_preparation(data_directory, MAXIMUM_DATA_NUM)
print('Finished prepare data.')
#runs the training of the attentional RNN
#train_RNN(SEQUENCE_LENGTH, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, KEEP_PROB, BATCH_SIZE, NUM_EPOCHS, DELTA, LEARNING_RATE, ALPHA_DIVIDER, MAXIMUM_DATA_NUM)
print('Finished train RNN.')
#runs the training of the LSTM 
train_LSTM(MAX_SUMMARY_LEN, MAX_TEXT_LEN, D, LSTM_HIDDEN_SIZE, LEARNING_RATE, K, TRAINING_ITERS, MAXIMUM_DATA_NUM)
print('Finished training.')
