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


filename = 'projection_edge_list.txt'

#By using with open... the file will automatically close outside the statement, so you don't need to manually close it
with open(filename, 'r') as f:# Open the file with read only ('r') permissions 
    lines = f.read().splitlines()  # Reads in the file line by line, saves each line as an element in a list
    
# Each line of the edge list is formated as node1 node2 with the separating character just a space
# With this format, we can read the edge list directedly using read_edgelist
G = nx.read_edgelist(filename, create_using = nx.Graph, nodetype = int)
    #We specify using create_using what type of graph to make, and nodetype what we should store the nodes as (i.e. int, float, str)

#Let's see how big this network is
print('\nNumber of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

centrality = nx.eigenvector_centrality(G)
#print('Centrality:')
#print(['%s %0.2f'%(node,centrality[node]) for node in centrality])


# In[16]:


plt.figure(figsize = (16,9))
nbins=20
plt.hist(eigen_your_mom.values(),bins=nbins,log=True)
plt.title("Log Plot - Eigenvector Centrality Distribution")


# In[24]:


deg_cen=nx.degree_centrality(G)
print('Deg Centrality:')
print(['%s %0.2f'%(node,deg_cen[node]) for node in deg_cen])


# In[18]:


matrix=nx.adjacency_matrix(G)
spectrum=nx.adjacency_spectrum(G)
#print(matrix)
print(spectrum)


# In[20]:


plt.figure(figsize = (16,9))
plt.hist(centrality.values())
plt.title("Eigenvector Centrality Distribution")
#plt.xlabel("")
#plt.ylabel("", fontsize = 100)


# In[3]:



dict_comic=dict(sorted(centrality.items(), key = lambda x: x[1], reverse = True))

f = open("marvel_eigenvector_centralities.txt","w")

f.write( str(dict_comic) )

f.close()


# In[20]:


#####RUNNING KATZ#####
comic_katz=nx.katz_centrality(G, alpha=1/247, beta=1.0, max_iter=100000)


# In[21]:


dict_katz_comic=dict(sorted(comic_katz.items(), key = lambda x: x[1], reverse = True))
print(dict_katz_comic)


# In[22]:


f4 = open("comic_katz.txt","w")

f4.write( str(dict_katz_comic) )

f4.close()


# In[8]:


with open('marvel_between_centralities.txt') as f:
    lines = f.readlines()


# In[14]:


plt.figure(figsize = (16,9))
plt.hist(between_your_mom.values())
plt.title("Betweenness Centrality Distribution")


# In[12]:


plt.figure(figsize = (16,9))
nbins=20
plt.hist(between_your_mom.values(),bins=nbins,log=True)
plt.title("Log Plot - Betweenness Centrality Distribution")


# In[3]:


pos = nx.spring_layout(G, k=0.15, iterations=20, seed=675)
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

#fig, ax = plt.subplots(figsize = (20, 20))
#draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')


# In[23]:


fig, ax = plt.subplots(figsize = (20, 20))
draw(G, pos, dict_katz_comic, 'Katz Centrality')


# In[12]:


f2 = open("marvel_between_centralities.txt","w")

f2.write( str(dic_bet_comic) )

f2.close()


# In[21]:


plt.figure(figsize = (16,9))
nbins=20
#hist, bins, _ = plt.hist(centrality.values(), bins=8)
#logbins=np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(centrality.values(),bins=nbins,log=True)
plt.title("Log-Log Plot - Eigenvector Centrality Distribution")
#plt.xscale('log')
#plt.loglog(basex=10,basey=10)


# In[48]:


centrality_sequence = np.array(list(centrality.values())) #Convert it to a numpy array so we can get the counts
centralities, counts = np.unique(centrality_sequence, return_counts=True)
fraction = counts / sum(counts) #normalize the counts to get the fraction of nodes

fig, ax = plt.subplots(figsize = (8, 6)) #Create 6 in x 8 in figure
ax.scatter(np.log(centralities), np.log(fraction)) #Plot the log-log plot as a scatter plot
ax.set_title('Log-Log Distribution Plot', fontsize = 20)
ax.set_ylabel(r"Log of Fraction of eigenvector centralities)", fontsize=15) #You can add LaTeX to text using $$ and the r in front of the string
ax.set_xlabel(r"Log of Eigenvector Centralities)", fontsize=15)


# In[6]:


filename = 'just_comic_vertices.txt'

#By using with open... the file will automatically close outside the statement, so you don't need to manually close it
with open(filename, 'r') as f: # Open the file with read only ('r') permissions 
    lines = f.read().splitlines()  # Reads in the file line by line, saves each line as an element in a list
    
# Each line of the edge list is formated as node1 node2 with the separating character just a space
# With this format, we can read the edge list directedly using read_edgelist
G_1 = nx.read_edgelist(filename, create_using = nx.Graph, nodetype = int)


# In[1]:


with open("just_comic_vertices.txt", "r") as f:
    score_strings=[]
    for line in f:
        if line[:-1]!='':
            score_strings.append(line[:-1])
lst=[]
score_ints=list(map(int,score_strings))
score_ints.count(6769)


# In[2]:


print(lst)
print("your mom")
print(score_ints)


# In[5]:


import pandas as pd
import sklearn.preprocessing

file_name = 'MarvelMovies edgelist.xlsx'

MCU_df = pd.read_excel(file_name, header = None)
le = sklearn.preprocessing.LabelEncoder()
le.fit(list(MCU_df[0]) + list(MCU_df[1]))
list1 = le.transform(MCU_df[0])
list2 = le.transform(MCU_df[1])
edges = list(zip(list1, list2))


# In[6]:


import itertools

G1 = nx.Graph()
for node_tuple in edges:
    G1.add_edges_from(itertools.combinations(node_tuple, 2))
#nx.draw(G1)
#plt.show()
#pos = nx.spring_layout(G)


# In[6]:


movie_eig_cen=nx.eigenvector_centrality(G1)
dict_movie=dict(sorted(movie_eig_cen.items(), key = lambda x: x[1], reverse = True))

f1 = open("cinema_eigenvector_centralities.txt","w")

# write file
f1.write( str(dict_movie) )

# close file
f1.close()


# In[17]:


betweeness_movie=nx.betweenness_centrality(G1)
dict_bet_movie=dict(sorted(betweeness_movie.items(), key = lambda x: x[1], reverse = True))

f3 = open("cinema_between_centralities.txt","w")

f3.write( str(dict_bet_movie) )

f3.close()


# In[5]:


L = nx.normalized_laplacian_matrix(G1)
e = numpy.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))


# In[7]:


katz_movie=nx.katz_centrality(G1, alpha=1/1.249, beta=1.0, max_iter=150000000, tol=1.00000000001)


# In[9]:


print(katz_movie)


# In[17]:


pos = nx.spring_layout(G1, k=0.15, iterations=20, seed=675)
def draw(G1, pos, measures, measure_name):
    
    nodes = nx.draw_networkx_nodes(G1, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    #labels = nx.draw_networkx_labels(G,pos,labels,font_size=16,font_color='r')
    edges = nx.draw_networkx_edges(G1, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

fig, ax = plt.subplots(figsize = (20, 20))
draw(G1, pos, katz_movie, 'Katz Centrality')


# In[20]:


#katz_movie=nx.katz_centrality(G1, alpha=1/1.249, beta=1.0, max_iter=100000, tol=1e-03)
dict_katz_movie=dict(sorted(katz_movie.items(), key = lambda x: x[1], reverse = True))

f5 = open("movie_katz.txt","w")

f5.write( str(dict_katz_movie) )

f5.close()


# In[18]:


fig, ax = plt.subplots(figsize = (20, 20))
draw(G1, pos, nx.betweenness_centrality(G1), 'Betweenness Centrality')


# In[48]:


plt.figure(figsize = (16,9))
nbins=20
plt.hist(movie_eig_cen.values(),bins=nbins,log=True)
plt.title("Log-Log Plot - Eigenvector Centrality Distribution")


# In[49]:


plt.figure(figsize = (16,9))
plt.hist(movie_eig_cen.values())
plt.title("Eigenvector Centrality Distribution")

