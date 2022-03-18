#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required packages
import powerlaw
import networkx as nx #networks package we will use for demos
import matplotlib.pyplot as plt #networkx uses matplotlib to generate its plots
from matplotlib import figure

import numpy as np #this is the linear algebra package for working with matrices
import pandas as pd #pandas is a dataframe package that is useful for managing network attributes

import math #use this package to take the log of a scalar, use numpy to take the element-wise log of an array


# In[2]:


filename = 'projection_edge_list.txt'

#By using with open... the file will automatically close outside the statement, so you don't need to manually close it
with open(filename, 'r') as f: # Open the file with read only ('r') permissions 
    lines = f.read().splitlines()  # Reads in the file line by line, saves each line as an element in a list
    
# Each line of the edge list is formated as node1 node2 with the separating character just a space
# With this format, we can read the edge list directedly using read_edgelist
G = nx.read_edgelist(filename, create_using = nx.Graph, nodetype = int)
    #We specify using create_using what type of graph to make, and nodetype what we should store the nodes as (i.e. int, float, str)

#Let's see how big this network is
print('\nNumber of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

centrality = nx.eigenvector_centrality(G)
print('Centrality:')
print(['%s %0.2f'%(node,centrality[node]) for node in centrality])


# In[57]:


# #What the the network look like? Maybe too big to really see clearly
nx.draw(G)
plt.show()

#Let's make the nodes smaller and arrange them with less overlapping and try again
pos = nx.spring_layout(G) #Uses the Fruchterman-Reingold force-directed algorith, to position nodes and minimize overlapping
#Note: for larger graphs, these positioning algorithms can be quite slow

#We can also make the figure bigger so the graph is less squished
#fig, ax = plt.subplots(figsize = (24, 24)) #Create 12 inches x 12 inches figure. We include ax = ax in nx.draw to draw on this figure


# In[56]:


dict(sorted(centrality.items(), key = lambda x: x[1], reverse = False))


# In[29]:


plt.figure(figsize = (16,9))
plt.hist(centrality.values())
#plt.title("")
#plt.xlabel("")
#plt.ylabel("", fontsize = 100)


# In[53]:


plt.figure(figsize = (16,9))
nbins=20
#hist, bins, _ = plt.hist(centrality.values(), bins=8)
#logbins=np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(centrality.values(),bins=nbins,log=True)
#plt.xscale('log')
#plt.loglog(basex=10,basey=10)


# In[48]:


centrality_sequence = np.array(list(centrality.values())) #Convert it to a numpy array so we can get the counts
centralities, counts = np.unique(centrality_sequence, return_counts=True)
fraction = counts / sum(counts) #normalize the counts to get the fraction of nodes

fig, ax = plt.subplots(figsize = (8, 6)) #Create 6 in x 8 in figure
ax.scatter(np.log(centralities), np.log(fraction)) #Plot the log-log plot as a scatter plot
ax.set_title('Log-Log Distribution Plot', fontsize = 20)
ax.set_ylabel(r"Log of Fraction of eigenvector centralities", fontsize=15) #You can add LaTeX to text using $$ and the r in front of the string
ax.set_xlabel(r"Log of Eigenvector Centralities", fontsize=15)


# In[5]:


results = powerlaw.Fit(list(centrality.values()))
print(results.power_law.alpha)
print(results.power_law.xmin)
R, p = results.distribution_compare('power_law', 'lognormal')


# In[3]:


bet = nx.betweenness_centrality(G)
results1 = powerlaw.Fit(list(bet.values()))
print(results1.power_law.alpha)
print(results1.power_law.xmin)
R1, p1 = results1.distribution_compare('power_law', 'lognormal')


# In[4]:


file1 = open('edward_comic_katz.txt')
lst1 = [] 
for line in file1:
    lst1.append(float(line))
temp1 = list(enumerate(lst1))
katz = dict(temp1)
results2 = powerlaw.Fit(list(katz.values()))
print(results2.power_law.alpha)
print(results2.power_law.xmin)
R2, p2 = results2.distribution_compare('power_law', 'lognormal')


# In[5]:


R1


# In[6]:


R2


# In[7]:


powerlaw.plot_pdf(list(centrality.values()), color = 'b')


# In[35]:


fig2 = results.plot_pdf(color = 'b', linewidth=2)
results.power_law.plot_pdf(color = 'b', linestyle= '--', ax=fig2)
results.plot_ccdf(color= 'r', linewidth=2, ax=fig2)
results.power_law.plot_ccdf(color= 'r', linestyle='--', ax=fig2)
plt.xlabel("Eigenvector Centrality Frequency")
plt.ylabel("$p(X)$")
plt.title("Distribution pdf vs ccdf")


# In[8]:


fig21 = results1.plot_pdf(color = 'b', linewidth=2)
results1.power_law.plot_pdf(color = 'b', linestyle= '--', ax=fig21)
results1.plot_ccdf(color= 'r', linewidth=2, ax=fig21)
results1.power_law.plot_ccdf(color= 'r', linestyle='--', ax=fig21)
plt.xlabel("Betweenness Centrality Frequency")
plt.ylabel("$p(X)$")
plt.title("Distribution pdf vs ccdf")


# In[10]:


fig22 = results2.plot_pdf(color = 'b', linewidth=2)
results2.power_law.plot_pdf(color = 'b', linestyle= '--', ax=fig22)
results2.plot_ccdf(color= 'r', linewidth=2, ax=fig22)
results2.power_law.plot_ccdf(color= 'r', linestyle='--', ax=fig22)
plt.xlabel("Katz Centrality Frequency")
plt.ylabel("$p(X)$")
plt.title("Distribution pdf vs ccdf")


# In[13]:


fig4 = results2.plot_ccdf(linewidth=3)
results2.power_law.plot_ccdf(ax=fig4, color='r', linestyle='--')
results2.lognormal.plot_ccdf(ax=fig4, color='g', linestyle='--')
plt.xlabel("Katz Centrality Frequency")
plt.ylabel("$p(X)$")
plt.title("Power law vs log normal fit")


# In[11]:


fig41 = results1.plot_ccdf(linewidth=3)
results1.power_law.plot_ccdf(ax=fig41, color='r', linestyle='--')
results1.lognormal.plot_ccdf(ax=fig41, color='g', linestyle='--')
plt.xlabel("Betweenness Centrality Frequency")
plt.ylabel("$p(X)$")
plt.title("Power law vs log normal fit")


# In[11]:


fig42 = results2.plot_ccdf(linewidth=3)
results2.power_law.plot_ccdf(ax=fig42, color='r', linestyle='--')
results2.lognormal.plot_ccdf(ax=fig42, color='g', linestyle='--')
plt.xlabel("Eigenvector Centrality Frequency")
plt.ylabel("$p(X)$")
plt.title("Power law vs log normal fit")


# In[25]:


with open("just_comic_vertices.txt", "r") as f:
    score_strings = []
    for line in f:
        if line[:-1] != '':
            score_strings.append(line[:-1])

lst = []
score_ints = list(map(int, score_strings))
score_ints.count(11877)
len(score_ints)


# In[8]:


x, y = results.cdf()
bin_edges, probability = results.pdf()
y = results.lognormal.cdf(data = [300, 350])
y = results.lognormal.pdf()


# In[9]:


print(x)


# In[10]:


print(y)


# In[11]:


R, p = results.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print(R, p)


# In[15]:


a = [x for x in range(6485)]
a.isin(3)

