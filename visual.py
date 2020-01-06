# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, marker = 'o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy = (xs[i], ys[i]))

def create_PCA(word_vector_list, compo = 2):
    pca = PCA(n_components=compo)
    xys = pca.fit_transform(word_vector_list)
    xs = xys[:,0]
    ys = xys[:,1]
    
    return xys, xs, ys

def create_PCA_three(word_vector_list, compo = 3):
    pca = PCA(n_components=compo)
    xyzs = pca.fit_transform(word_vector_list)
    xs = xyzs[:,0]
    ys = xyzs[:,1]
    zs = xyzs[:,2]
    
    return xyzs, xs, ys, zs


