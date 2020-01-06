# -*- coding: utf-8 -*-
#test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
#import preprocessor as pre
import model
import visual
import datetime
from sklearn.decomposition import PCA
import os
import re
import nltk
from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
from sklearn.cluster import AgglomerativeClustering, DBSCAN



# Path and Input
path = "./data/"
in_file = "abs_eng_1210.txt"

f = open(os.path.join(path,in_file), 'r', encoding = 'utf-8')
wv = f.readlines()

retokenize = nltk.RegexpTokenizer("[a-zA-Z]+")

wv_input = []

for line in wv:
    wv_input.append(retokenize.tokenize(line))

skip_gram_model = model.build_sg_model(sentences = wv_input,
                                       vocab_size = 100, #dimension size
                                       window_size = 4,
                                       min_count=5,
                                       workers=4)

skip_gram_model.init_sims(replace=True)

word_vector = skip_gram_model.wv
vocabs = word_vector.vocab.keys()
word_vector_list = [word_vector[v] for v in vocabs]

#Save Result
now = datetime.datetime.now()
now_date = now.strftime("%Y_%m_%d_%H_%M_%S")
output_path = "./output/skip_gram_"+now_date+".model"
skip_gram_model.save(output_path)
#model.intersect_word2vec_format(fname=file_name, binary=True)

#Get PCA values of the vector(dim = 2)
xys, xs, ys = visual.create_PCA(word_vector_list)
xyzs, xs_3, ys_3, zs_3 = visual.create_PCA_three(word_vector_list)
#Viuallize
#visual.plot_2d_graph(vocabs, xs, ys)

#Get Word2Vec Dictionary(Original, PCA)
dict_skip_gram = {}
dict_skip_gram_PCA = {}
dict_skip_gram_PCA_3 = {}

for k, v in zip(word_vector.index2word, word_vector.vectors):
    k_str = str(k)
    dict_skip_gram[k_str] = v   

for k,v in zip(word_vector.index2word, xys):
    k_str = str(k)
    dict_skip_gram_PCA[k_str] = v

for k,v in zip(word_vector.index2word, xyzs):
    k_str = str(k)
    dict_skip_gram_PCA_3[k_str] = v
    
# Cluster Definition
clusterer_eu = KMeansClusterer(10, euclidean_distance, repeats=10)
clusterer_co = KMeansClusterer(10, cosine_distance, repeats=10)

# PCA Values
clu_input = dict_skip_gram_PCA.values()

# Cluster Assign Result
assigned_eu = clusterer_eu.cluster(clu_input, assign_clusters=True)
assigned_co = clusterer_co.cluster(clu_input, assign_clusters=True)

clu_dict_eu = {}
for clu, key in zip(assigned_eu, dict_skip_gram_PCA.keys()):
        clu_dict_eu.setdefault(clu, []).append(key)
        
clu_dict_co = {}
for clu, key in zip(assigned_co, dict_skip_gram_PCA.keys()):
        clu_dict_co.setdefault(clu, []).append(key)
        

'''
cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
#
small_input = []
cluster_input = []
for v in dict_skip_gram_PCA.values():
    for k in v:
        small_input.append(k)
    cluster_input.append(small_input)
    small_input = []

arr_clu = np.asarray(cluster_input)
k_out = cluster.fit_predict(cluster_input)
'''

#plt.figure(figsize=(10, 7))
#plt.scatter(arr_clu[:,0], arr_clu[:,1], c=cluster.labels_, cmap='rainbow')

