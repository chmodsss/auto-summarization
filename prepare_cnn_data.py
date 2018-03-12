# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:27:24 2018
mixture of 
tutorial for preparing and cleaning our dataset, source:
https://machinelearningmastery.com/prepare-news-articles-text-summarization/
and 
https://github.com/JRC1995/Abstractive-Summarization/blob/master/Data%20Pre-processing.ipynb
@author: kloe_mr
"""
    
#%%
from __future__ import division
import numpy as np

def data_preparation(directory, MAXIMUM_DATA_NUM):
    #filename = 'glove.6B.50d.txt' 
    filename = '/home/kloe_mr/AnacondaProjects/tf-rnn-attention/JRC1995/glove.6B.50d.txt'
    
    # (glove data set from: https://nlp.stanford.edu/projects/glove/)
    # load preembedded GloVe words from wiki-articles (400K words with dim 50)
    def loadGloVe(filename):
        vocab = []
        embd = []
#        vocab.append('eos') #TRY
#        embd.append(0)
#        vocab.append('unk')
#        embd.append(1)
        try:        ##ADDED MKLOENNE
            file = open(filename, encoding='utf-8')
            print(file)
            for line in file.readlines(): 
                row = line.strip().split(' ')
                vocab.append(row[0])
                embd.append(row[1:])
            print('DATA-PREPARATION: GloVe Loaded.')
        finally:    ##ADDED MKLOENNE
            file.close()
        return vocab,embd
    
    # Pre-trained GloVe embedding
    vocab,embd = loadGloVe(filename)
    # convert list of embd to array
    embedding = np.asarray(embd)
    embedding = embedding.astype(np.float32)
    
    word_vec_dim = len(embd[0]) # word_vec_dim = dimension of each word vectors
            
    #%% DATASET CNN NEWS ARTICLES (MACHINELEARNINGMASTERY)
    
    from os import listdir
    import string
       
    # load doc into memory
    def load_doc(filename):
        # open the file as read only
        file = open(filename, encoding='utf-8')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text
    
    # split a document into news story and highlights
    def split_story(doc):
        # find first highlight
        index = doc.find('@highlight')
        # split into story and highlights
        story, highlights = doc[:index].split(), doc[index:].split() #.split('@highlight') ##CHANGED MKLOENNE
        # strip extra white space around each highlight
        highlights = [h.strip() for h in highlights if len(h) > 0]
        return story, highlights
    
    # load all stories in a directory
    def load_stories(directory):
        stories = list()
        articles = list()      ## ADDED BY MKLOENNE
        summaries = list()     ## ADDED BY MKLOENNE
        for name in listdir(directory):
            filename = directory + '/' + name
            # load document
            doc = load_doc(filename)
            # split into story and highlights
            story, highlights = split_story(doc)
            # store
            stories.append({'story':story, 'highlights':highlights})
            articles.append(story)    ## ADDED BY MKLOENNE 
            summaries.append(highlights)      ## ADDED BY MKLOENNE        
        return stories, articles, summaries    ## CHANGED BY MKLOENNE
    
    
    # clean a list of lines
    def clean_lines(lines):
        cleaned = list()
        # prepare a translation table to remove punctuation
        table = str.maketrans('', '', string.punctuation)
        for line in lines:
            """
            # replace contraction words: ##ADDED BY MKLOENNE 
            if word in contractions:
                line = line.replace(line,contractions[line])
            # tokenize on white space
            line = line.split(' ') 
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # remove stopwords (ADDED BY MKLOENNE)
            # line = [word for word in line if word not in stopwords.words('english')] 
            """
            # strip source cnn office if it exists
            index = line.find('(CNN) -- ')
            if index > -1:
                line = line[index+len('(CNN)'):]
            # remove @highlights
            index = line.find('@highlight')
            if index > -1:
                line = line.replace('@highlight','')                   
            # remove line with Copyright (ADDED BY MKLOENNE) 
            if line.startswith('Copyright'): 
                continue
            # convert to lower case
            line = line.lower() 
            # remove punctuation from each token
            line = line.translate(table)
           # store as string
            cleaned.append(line)
        # remove empty strings
        cleaned = [c for c in cleaned if len(c) > 0]
        return cleaned
    ## tokenize could be a problem try without tokenizing and stopwords 

    stories, articles, summariesOld = load_stories(directory)
    print('DATA-PREPARATION: Loaded Stories %d' % len(stories))
    
    ## clean articles ## ADDED BY MKLOENNE
    texts = list()
    for text in articles:
        text = clean_lines(text)
        texts.append(text)
    
    ## clean summaries ## ADDED BY MKLOENNE
    summaries = list()
    for summary in summariesOld: 
        summary = clean_lines(summary)
        summaries.append(summary)
    
    print('DATA-PREPARATION: Texts & summaries cleaned')
    
    
    #%%
    
    #import random
    
    #index = random.randint(0,len(texts)-1)
    
    #print("SAMPLE CLEANED & TOKENIZED TEXT: \n\n"+str(texts[index]))        ##CHANGED PY2 TO PY3
    #print("\nSAMPLE CLEANED & TOKENIZED SUMMARY: \n\n"+str(summaries[index]))       ##CHANGED PY2 TO PY3
    
    #%%
    
    def np_nearest_neighbour(x):
        #returns array in embedding that's most similar (in terms of cosine similarity) to x
            
        xdoty = np.multiply(embedding,x)
        xdoty = np.sum(xdoty,1)
        xlen = np.square(x)
        xlen = np.sum(xlen,0)
        xlen = np.sqrt(xlen)
        ylen = np.square(embedding)
        ylen = np.sum(ylen,1)
        ylen = np.sqrt(ylen)
        xlenylen = np.multiply(xlen,ylen)
        cosine_similarities = np.divide(xdoty,xlenylen)
    
        return embedding[np.argmax(cosine_similarities)]
              
    def word2vec(word):  # converts a given word into its vector representation
        if word in vocab:
            return embedding[vocab.index(word)]
        else:
            return embedding[vocab.index('unk')]
    
    def vec2word(vec):   # converts a given vector representation into the represented word 
        for x in range(0, len(embedding)):     ##CHANGED PY2 TO PY3
                if np.array_equal(embedding[x],np.asarray(vec)):
                    return vocab[x]
        return vec2word(np_nearest_neighbour(np.asarray(vec)))
    
    #%%
    # reduce data to given maximum data size
    texts = texts[0:MAXIMUM_DATA_NUM]
    summaries = summaries[0:MAXIMUM_DATA_NUM]
    
    print("DATA-PREPARATION: Data reduced to " + str(MAXIMUM_DATA_NUM) + ' texts/summaries')
    
    #%%
    
    vocab_limit = vocab[:]
    # add words from texts to the vocabulary list
    def words2vocab(texts, vocab):
        for text in texts:
            for word in text:
                if word not in vocab:
                    vocab.append(word)
        print("DATA-PREPARATION: vocab_limit added by" + str(texts))
     
    # add embedding of words from text to the embedded vocabulary list 
    def embds2vocab(texts, embd_vocab):
        for text in texts:
            for word in text:
                if word not in vocab:
                    embd.append(word2vec(word))
        print("DATA-PREPARATION: embd_limit added by" + str(texts))
        
    words2vocab(texts, vocab_limit)
    words2vocab(summaries, vocab_limit)
    embds2vocab(texts, embd)
    embds2vocab(summaries, embd)
    #%% apply unk, eos, zeros to the vocabulary & its embedding
         
    embd_limit = list(embd)
               
#    if 'eos' not in vocab_limit:
#        vocab_limit.append('eos')
#        embd_limit.append(word2vec('eos'))
#    if 'unk' not in vocab_limit:
#        vocab_limit.append('unk')
#        embd_limit.append(word2vec('unk'))
    
    null_vector = np.zeros([word_vec_dim])
    
    vocab_limit.append('<PAD>')
    embd_limit.append(null_vector)
    print("DATA-PREPARATION: applied unk, eos, zeros")
    
    
    #%%
    # embedd the words of the texts 
    def append_w2v(texts):    
        vecs = []
        for text in texts:
            vec = []
            for word in text: 
                vec.append(word2vec(word))
            vec.append(word2vec('eos'))
            vec = np.asarray(vec)
            vec = vec.astype(np.float32)
            vecs.append(vec)
        print("DATA-PREPARATION: embedded & prepared " + str(texts))
        return vecs
    
    vec_summaries = append_w2v(summaries)
    vec_texts = append_w2v(texts)
    
    #%% 
        
    # Saving processed data in another file.
    print(len(vec_summaries))
    import pickle
    with open('vocab_limit', 'wb') as fp:
        pickle.dump(vocab_limit, fp)
    with open('embd_limit', 'wb') as fp:
        pickle.dump(embd_limit, fp)
    with open('vec_summaries', 'wb') as fp:
        pickle.dump(vec_summaries, fp)
    with open('vec_texts', 'wb') as fp:
        pickle.dump(vec_texts, fp)
    with open('raw_texts', 'wb') as fp: 
        pickle.dump(texts, fp)
    with open('raw_sums', 'wb') as fp: 
        pickle.dump(summaries, fp)
    

    
    #%%
    # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    '''
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
    }
    '''