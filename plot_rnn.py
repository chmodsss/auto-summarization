# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:10:37 2018

@author: kloe_mr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cbook as cbook

#CHANGE PATH AND NAMES FOR EACH FILES!
fname = cbook.get_sample_data('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/arnn_test_150_batches6_epochs.csv', asfileobj=False)

# plot precision and recall
plt.plotfile(fname, (0, 3, 4))
plt.savefig('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/exp1_prec_rec_rnn_test.png')

# plott f_measure 
plt.plotfile(fname, (0, 5))
plt.savefig('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/exp1_f_measure_rnn_test.png')



# read from csv into record array
dataset = pd.read_csv('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/arnn_test_150_batches6_epochs.csv')# in your case right name of your file
X=dataset.iloc[:,:-1].values  #  this will convert dataframe to object
df = pd.DataFrame(X)
df = df.drop([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], axis=1)
# calc means on columns
ans = np.mean(df, axis=0)


#6-15
'''
d = pd.read_csv("H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/arnn_train_125_batches3_epochs.csv", nrows=1, usecols=[6,7,8,9,10,11,12,13,14,15])
legend = ['runtime/sec', 'embedding_dim','hidden_size','attention_size','keep_prob','batch_size','epochs','delta','learning_rate','alpha_div']
runtime_sec = d['RUNTIME/sec']
embedding_dim = d['EMBEDDING_DIM']
hidden_size = d['HIDDEN_SIZE']      
attention_size = d['ATTENTION_SIZE']
keep_prob = d['KEEP_PROB']
batch_size = d['BATCH_SIZE']
epochs = d['EPOCHS']
delta = d['DELTA']
learning_rate = d['LEARNING_RATE']
alpha_div = d['ALPHA_DIVIDER']


d2 = pd.read_csv("H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/lstm_train.csv", nrows=1, usecols=[6,7,8,9,10,11,12])
legend = ['runtime/sec','hidden_size','K','D','epochs','learning_rate']
runtime_sec_lstm = d2['RUNTIME/sec']
hidden_size_lstm = d2['HIDDEN_SIZE']
K = d2['K']
D = d2['D']
epochs_lstm = d2['EPOCHS']
learning_rate_lstm = d2['LEARNING_RATE']

rnn_params = (embedding_dim, hidden_size, attention_size, batch_size, alpha_div, epochs)

ind = np.arange(len(rnn_params))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, rnn_params, width,
                color='SkyBlue', label='RNN')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_title('Hyperparameters')
ax.set_xticks(ind)
ax.set_xticklabels(('EMBD_DIM', 'HIDDEN', 'ATT_SIZE', 'BATCHES', 'ALPHA_DIV', 'EPOCHS'))
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.95}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.0*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
#autolabel(rects2, "right")
#plt.show()
plt.savefig('exp1_rnn_params1.png')

rnn_params = (keep_prob, delta, learning_rate)
#lstm_params = (25, 32, 34, 20, 25)

ind = np.arange(len(rnn_params))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, rnn_params, width,
                color='SkyBlue', label='RNN')
#rects2 = ax.bar(ind + width/2, lstm_params, width,
#                color='IndianRed', label='LSTM')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_title('Hyperparameters')
ax.set_xticks(ind)
ax.set_xticklabels(('KEEP_PROB', 'DELTA', 'LEARNING_RATE'))
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.95}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.0*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
#autolabel(rects2, "right")
#plt.show()
plt.savefig('exp1_rnn_params2.png')
'''