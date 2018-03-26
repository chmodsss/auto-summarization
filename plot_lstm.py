# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:59:05 2018

@author: kloe_mr
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cbook as cbook

#CHANGE PATH AND NAMES FOR EACH FILES!
fname = cbook.get_sample_data('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/lstm_train.csv', asfileobj=False)

# plot precision and recall
plt.plotfile(fname, (0, 3, 4))
plt.savefig('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/exp1_prec_rec_lstm_train.png')

# plott f_measure 
plt.plotfile(fname, (0, 5))
plt.savefig('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/exp1_f_measure_lstm_train.png')


# read from csv into record array
dataset = pd.read_csv('H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles_5/lstm_train.csv')# in your case right name of your file
X=dataset.iloc[:,:-1].values  #  this will convert dataframe to object
df = pd.DataFrame(X)
df = df.drop([0, 1, 2, 6, 7, 8, 9, 10, 11], axis=1)
# calc means on columns
ans = np.mean(df, axis=0)

'''
d2 = pd.read_csv("H:/AnacondaProjects/tf-rnn-attention/Bachelorarbeit/BA/experiments/1000articles/lstm_train.csv", nrows=1, usecols=[6,7,8,9,10,11,12])
legend = ['runtime/sec','hidden_size','K','D','epochs','learning_rate']
runtime_sec_lstm = d2['RUNTIME/sec']
hidden_size_lstm = d2['HIDDEN_SIZE']
K = d2['K']
D = d2['D']
epochs_lstm = d2['EPOCHS']
learning_rate_lstm = d2['LEARNING_RATE']

lstm_params = (hidden_size_lstm, epochs_lstm ,K ,D, learning_rate_lstm)

ind = np.arange(len(lstm_params))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, lstm_params, width,
                color='LightGreen', label='LSTM')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_title('Hyperparameters')
ax.set_xticks(ind)
ax.set_xticklabels(('HIDDEN', 'EPOCHS', 'K', 'D', 'LEARNING_RATE'))
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
#plt.savefig('exp1_lstm_params.png')

'''