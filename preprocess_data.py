"""
Description: contains code for
(1) preprocessing the labeled dataset file,
(2) creating 10 different sets of train/test/dev files, one set for each dimension
(3) creating BERT-friendly dataset files
"""
import re
import os
import numpy as np
import sys
from os.path import join
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.tokenize import TweetTokenizer
tokenize = TweetTokenizer().tokenize

def preprocessText(text):
    """
    A function that applies basic preprocessing to a given piece of text using regular expressions
    :param text: the input string to be preprocessed
    :return: a cleaned version of the same text
    """
    # text = re.sub(pattern=r'\b[A-Za-z.\s]+:', string=text, repl=' ')  # remove characters - for round3
    text = re.sub(pattern=r'\<(.*?)\>', string=text, repl=' ')  # remove html tags - for round4
    text = re.sub(pattern=r'/?&gt;', string=text, repl=' ')  # remove html tags - for round4
    text = text.replace('"', '').strip()
    text = ' '.join(tokenize(text))
    return text

def splitToTrainTest():
    """
    The provided data/labeled-dataset.tsv file contains annotation results for each sentence across all dimensions.
    This code turns the dataset into a form that allows for training binary classifiers for each of the 10 dimensions.
    For each dimension, a sentence is positive if 2 or more out of 3 agreed that a dimension exists,
    and negative if 0 out of the 3 said a dimension exists.
    :return: train/test/dev.tsv files stored for each dimension.
    """

    # read dataset file
    df = pd.read_csv('data/labeled-dataset.tsv', sep='\t')
    # select text column to use
    text_col_idx = df.columns.tolist().index('h_text')

    # list the 10 dimensions
    dims = ['social_support',
            'conflict',
            'trust',
            'fun',
            'similarity',
            'identity',
            'respect',
            'romance',
            'knowledge',
            'power']

    for dim in dims:
        save_dir = 'data/%s'%dim # directory where to save the files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # select which column to use as target variable
        dim_col_idx = df.columns.tolist().index(dim)

        # append positive/negative samples to X and y lists
        X = []
        y = []
        for line in df.values:
            label = line[dim_col_idx]
            if (label >= 2) | (label == 0):
                text = preprocessText(line[text_col_idx])
                X.append(text)
                y.append(int(label >= 2))

        X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=42)
        X_t, X_v, y_t, y_v = train_test_split(X_t, y_t, test_size=0.5, random_state=42)

        # save to tsv files
        out = []
        for i,(X_,y_) in enumerate(zip(X_tr,y_tr)):
            out.append((y_,'a',X_))
        df2 = pd.DataFrame(out)
        df2.to_csv(join(save_dir,'train.tsv'),sep='\t',header=None)

        out = []
        for i,(X_,y_) in enumerate(zip(X_t,y_t)):
            out.append((y_,'a',X_))
        df2 = pd.DataFrame(out)
        df2.to_csv(join(save_dir,'test.tsv'),sep='\t',header=None)

        out = []
        for i,(X_,y_) in enumerate(zip(X_v,y_v)):
            out.append((y_,'a',X_))
        df2 = pd.DataFrame(out)
        df2.to_csv(join(save_dir,'dev.tsv'),sep='\t',header=None)
    return

def padBatch(list_of_list_of_arrays,max_seq=None):
    """
    A function that returns a numpy array that creates a B @ (seq x dim) sized-tensor
    :param list_of_list_of_arrays:
    :return:
    """

    if max_seq:
        list_of_list_of_arrays = [X[:max_seq] for X in list_of_list_of_arrays]


    # get max length
    mx = max(len(V) for V in list_of_list_of_arrays)

    # get array dimension by looking at the 1st sample
    array_dimensions = [len(V[0]) for V in list_of_list_of_arrays]
    assert min(array_dimensions)==max(array_dimensions), "Dimension sizes do not match within the samples!"
    dim_size = array_dimensions[0]

    # get empty array to put in
    dummy_arr = [0]*dim_size

    # create additional output
    out = []
    for V in list_of_list_of_arrays:
        V = V.tolist()
        out.append(V+[dummy_arr]*(mx-len(V)))

    # return
    return np.array(out)

def batchify(X, y, batch_size=50):
    """
    Given a list of data samples, returns them in a batch
    :param X:
    :param y:
    :param batch_size:
    :return:
    """
    out = []
    n_batches = int(np.ceil(len(X)/batch_size))
    for i in range(n_batches):
        X_b = X[i*batch_size:(i+1)*batch_size]
        y_b = y[i*batch_size:(i+1)*batch_size]
        out.append((X_b,y_b))

    return out

if __name__=='__main__':
    splitToTrainTest()