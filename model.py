# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec


def build_sg_model(sentences, vocab_size, window_size,min_count, workers):
    model = Word2Vec(sentences,
                     size = vocab_size,
                     window = window_size,
                     min_count = min_count,
                     workers = workers,
                     sg = 1)
    return model

def build_cbow_model(sentences, vocab_size, window_size,min_count, workers):
    model = Word2Vec(sentences,
                     size = vocab_size,
                     window = window_size,
                     min_count = min_count,
                     workers = workers,
                     sg = 0)
    return model

#rt_1 = retokenize.tokenize(wv[0])
#wt_1 = word_tokenize(wv[:2])