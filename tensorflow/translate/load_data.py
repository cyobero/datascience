#!/usr/bin

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

# Download and prepare the dataset 
# The dataset can be found at http://www.manythings.org/anki/
# It contains language translation (Spanish -> English) in the 
# following form:
#  
#    May I borrow this book? ?Puedo tomar prestado este libro?
#
# After diwbkiadubg the dataset, follow these steps:
#    1. Add a START and END token to each sentence.
#    2. Clean the sentences by removing special characters.
#    3. Create a word index and reverse word index (dictionaries mapping word --> id and id --> word).
#    4. Pad each sentence to a maximum length.

# Download the zip file
path_to_zip = tf.keras.utils.get_file(
         'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
          extract=True)

path_to_file = os.path.dirname(path_to_zip) + '/spa-eng/spa.txt'

# Convert Unicode file to ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
             if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the puncuation following it
    # e.g.: "he is a boy." --> "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r"\1", w)
    w = re.sub(r"[" "]+", " ", w)

    # replacing everything with space except(a-z, A-Z, ".". "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and end token to the sentence
    # so that model knows when to start and stop predicting
    w = '<start>' + w + ' <end>'
    return w


# 1. Remove accents
# 2. Clean sentences
# 3. Return word pairs in format: [ENGLISH, ESPANOL]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


