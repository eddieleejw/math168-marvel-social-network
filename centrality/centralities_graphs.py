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

from numpy import linalg as LA


# In[2]:


filename = 'projection_edge_list.txt'

#By using with open... the file will automatically close outside the statement, so you don't need to manually close it
with open(filename, 'r') as f: # Open the file with read only ('r') permissions 
    lines = f.read().splitlines()  # Reads in the file line by line, saves each line as an element in a list
    
# Each line of the edge list is formated as node1 node2 with the separating character just a space
# With this format, we can read the edge list directedly using read_edgelist
G = nx.read_edgelist(filename, create_using = nx.Graph, nodetype = int)
pos = nx.spring_layout(G, k=0.15, iterations=20, seed=675)


# In[3]:


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


# In[48]:


fig, ax = plt.subplots(figsize = (20, 20))
draw(G, pos, nx.degree_centrality(G), 'Degree Centrality')


# In[2]:


import pandas as pd
import sklearn.preprocessing

file_name = 'MarvelMovies edgelist.xlsx'

MCU_df = pd.read_excel(file_name, header = None)
le = sklearn.preprocessing.LabelEncoder()
le.fit(list(MCU_df[0]) + list(MCU_df[1]))
list1 = le.transform(MCU_df[0])
list2 = le.transform(MCU_df[1])
edges = list(zip(list1, list2))

import itertools

G1 = nx.Graph()
for node_tuple in edges:
    G1.add_edges_from(itertools.combinations(node_tuple, 2))

    

#fig, ax = plt.subplots(figsize = (20, 20))
#draw(G1, pos, nx.eigenvector_centrality(G1), 'Eigenvector Centrality')


# In[14]:


#nx.draw(G1)
#plt.show()

#Let's make the nodes smaller and arrange them with less overlapping and try again
pos = nx.spring_layout(G1) 

fig, ax = plt.subplots(figsize = (20, 20)) #Create 12 inches x 12 inches figure. We include ax = ax in nx.draw to draw on this figure
nx.draw(G1, pos)


# In[67]:


fig, ax = plt.subplots(figsize = (20, 20))
draw(G, pos, nx.eigenvector_centrality(G), 'Eigenvector Centrality')


# In[69]:


high_nodes=[858, 2663, 4897, 5715, 5305, 3804]
labels1 = {}    
for node in high_nodes:
    labels1[node] = node
fig, ax = plt.subplots(figsize = (20, 20))
nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, node_color = list(nx.eigenvector_centrality(G).values()),
                              nodelist=nx.eigenvector_centrality(G).keys())
nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
edges = nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G,pos,labels1,font_size=16,font_color='green')
plt.colorbar(nodes)


# In[7]:


#2 is value of max eigenvalue
draw(G, pos, nx.katz_centrality(G, alpha=1/2, beta=1.0, max_iter=100000, tol=1e-03), 'Katz Centrality')


# In[ ]:




