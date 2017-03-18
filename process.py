import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import re
import string

SEQ_LENGTH = 100

def read_text_data(filter_punc = True):

    with open("alice.txt", 'r') as infile:
        data = infile.read().lower()

    if filter_punc:

        data = data.translate(str.maketrans('','',string.punctuation))

    return data

def create_text_map(text_data):

    chars = sorted(list(set(text_data)))

    return {c:i for i, c in enumerate(chars)}

def summarize(text_data):

   char_dict = create_text_map(text_data)

   return len(text_data), char_dict

def create_sequences():

    text_data = read_text_data()

    n_chars, char_dict = summarize(text_data)


    dataX = []
    dataY = []
    for i in range(0, n_chars - SEQ_LENGTH, 1):

        seq_in = text_data[i:i+SEQ_LENGTH]
        seq_out = text_data[i +SEQ_LENGTH]
        dataX.append([char_dict[char] for char in seq_in])
        dataY.append(char_dict[seq_out])

    n_patterns = len(dataX)

    X = np.reshape(dataX, (n_patterns, SEQ_LENGTH, 1))
    X = X /  float(len(char_dict))


    y = np_utils.to_categorical(dataY)

    return X,y

def create_model():

    X,y = create_sequences()



    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')


    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, nb_epoch=20, batch_size=128, callbacks=callbacks_list)


create_model()
