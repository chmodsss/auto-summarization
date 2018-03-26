# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 23:35:27 2018

@author: kloe_mr
"""

from prepare_cnn_data import data_preparation
from seperated_autoencoder_LSTMS import train_LSTM

MAXIMUM_DATA_NUM = 1000
#data_directory = 'C:/Users/kloe_mr/Documents/data/cnn_stories/cnn/stories/'
data_directory = '/home/kloe_mr/AnacondaProjects/data/cnn/stories/'  ## for linux workstation 

#HYPERPARAMETERS FOR THE aRNN 
SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100
HIDDEN_SIZE = 250
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 50
#EXPLANATION' FOR BATCHES: https://theneuralperspective.com/2016/10/04/05-recurrent-neural-networks-rnn-part-1-basic-rnn-char-rnn/
NUM_EPOCHS = 3  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
LEARNING_RATE = 0.003
ALPHA_DIVIDER = 50  

#HYPERPARAMETERS FOR THE LSTMS
MAX_SUMMARY_LEN = 100
MAX_TEXT_LEN = 1000   
#D is a major hyperparameters. Windows size for local attention will be 2*D+1
D = 5
LSTM_HIDDEN_SIZE = 350#500
LEARNING_RATE = 0.003
K = 5#5
TRAINING_ITERS = 30 

#runs the data preprocessing 
#data_preparation(data_directory, MAXIMUM_DATA_NUM)
print('Finished prepare data.')
#runs the training of the LSTM 
train_LSTM(MAX_SUMMARY_LEN, MAX_TEXT_LEN, D, LSTM_HIDDEN_SIZE, LEARNING_RATE, K, TRAINING_ITERS, MAXIMUM_DATA_NUM)
print('Finished training.')
