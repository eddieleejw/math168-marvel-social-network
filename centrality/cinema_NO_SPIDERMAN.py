#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required packages

import networkx as nx #networks package we will use for demos
import matplotlib.pyplot as plt #networkx uses matplotlib to generate its plots
from matplotlib import figure
import matplotlib.colors as mcolors

import numpy as np #this is the linear algebra package for working with matrices
import pandas as pd #pandas is a dataframe package that is useful for managing network attributes

import math #use this package to take the log of a scalar, use numpy to take the element-wise log of an array
import numpy.linalg
#from numpy import linalg as LA


# In[2]:


def draw(G, pos, measures, measure_name):
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='r')
    edges = nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()


# In[3]:


import pandas as pd
import sklearn.preprocessing

file_name = 'no_spiderman_cinema.xlsx'

MCU_df = pd.read_excel(file_name, header = None)
le = sklearn.preprocessing.LabelEncoder()
le.fit(list(MCU_df[0]) + list(MCU_df[1]))
list1 = le.transform(MCU_df[0])
list2 = le.transform(MCU_df[1])
edges = list(zip(list1, list2))


# In[4]:


import itertools

G1 = nx.Graph()
for node_tuple in edges:
    G1.add_edges_from(itertools.combinations(node_tuple, 2))

pos = nx.spring_layout(G1)
fig, ax = plt.subplots(figsize = (12, 12))
nx.draw(G1)
plt.show()


# In[5]:


bet_no_spider=nx.betweenness_centrality(G1)
dict_bet_movie=dict(sorted(bet_no_spider.items(), key = lambda x: x[1], reverse = True))


# In[6]:


print(dict_bet_movie)


# In[6]:


pos = nx.spring_layout(G1, k=0.15, iterations=20, seed=675)
fig, ax = plt.subplots(figsize = (12, 12))
draw(G1, pos, nx.betweenness_centrality(G1), 'Marvel Cinema Katz Centrality - Without Spideman')


# In[19]:


eigen_no_spider=nx.eigenvector_centrality(G1)
dict_eigen_movie=dict(sorted(eigen_no_spider.items(), key = lambda x: x[1], reverse = True))
print(dict_eigen_movie)
fig, ax = plt.subplots(figsize = (12, 12))
draw(G1, pos, eigen_no_spider, 'Marvel Cinema Eigenvector Centrality - Without Spideman')


# In[12]:


L = nx.normalized_laplacian_matrix(G1)
e = numpy.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))


# In[16]:


katz_movie=nx.katz_centrality(G1, alpha=1/1.06, beta=1.0, max_iter=150000000, tol=1.00000000000001)


# In[20]:


#katz_no_spider=nx.eigenvector_centrality(G1)
dict_katz_movie=dict(sorted(katz_movie.items(), key = lambda x: x[1], reverse = True))
print(dict_katz_movie)
fig, ax = plt.subplots(figsize = (12, 12))
draw(G1, pos, katz_movie, 'Marvel Cinema Katz Centrality - Without Spideman')


# In[ ]:




