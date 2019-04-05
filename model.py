# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:33:14 2019

@author: autpx
"""

from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile

import numpy as np

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from numpy import argmax
from sklearn.metrics import confusion_matrix
import pandas as pd

np.random.seed(56) 

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['gold_label']
        s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
        if skip_no_majority and label == '-':
            continue
        yield (label, s1, s2)

def get_data(fn, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    print(max(len(x.split()) for x in left))
    print(max(len(x.split()) for x in right))

    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
    Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y

def predict_inference(model,text,hypothese):
    prepare_data_pred = lambda data: (to_seq(data[0]), to_seq(data[1]))
    coded=prepare_data_pred([[text],[hypothese]])
    results= model.predict([coded[0],coded[1]])

    inference=LABELS_INV[argmax(results)]
    print("Precicted Judgments: ",inference)
    return inference

#training_org = get_data('snli_1.0_train.jsonl')
#validation_org = get_data('snli_1.0_dev.jsonl')
test_org = get_data('snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training_org[0] + training_org[1])

VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
LABELS_INV = dict((v,k) for k,v in LABELS.items())
RNN = None
LAYERS = 1
USE_GLOVE = True
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 256
PATIENCE = 4 # 8
MAX_EPOCHS = 10
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'
#print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
#print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])
prepare_data_pred = lambda data: (to_seq(data[0]), to_seq(data[1]))

#training = prepare_data(training_org)
#validation = prepare_data(validation_org)
test = prepare_data(test_org)

print('Build model...')
print('Vocab size =', VOCAB)

GLOVE_STORE = 'precomputed_glove.weights'
if USE_GLOVE:
    if not os.path.exists(GLOVE_STORE + '.npy'):
        print('Computing GloVe')
        embeddings_index = {}
        f = open('glove.840B.300d.txt')
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                print('Missing from GloVe: {}'.format(word))

        np.save(GLOVE_STORE, embedding_matrix)

    print('Loading GloVe')
    embedding_matrix = np.load(GLOVE_STORE + '.npy')

    print('Total number of null word embeddings:')
    print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
else:
    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)

rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))

translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

prem = embed(premise)
hypo = embed(hypothesis)

prem = translate(prem)
hypo = translate(hypo)

if RNN and LAYERS > 1:
    for l in range(LAYERS - 1):
        rnn = RNN(return_sequences=True, **rnn_kwargs)
        prem = BatchNormalization()(rnn(prem))
        hypo = BatchNormalization()(rnn(hypo))
rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
prem = rnn(prem)
hypo = rnn(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)


model.summary()

#print('Training')
#_, tmpfn = tempfile.mkstemp()
#
#callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
#model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

model.load_weights(tmpfn)
#

y_test_predicted= predict_inference(model,test[0], test[1])
print (y_test_predicted)


