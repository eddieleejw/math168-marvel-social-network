#!/usr/bin/env python
# coding: utf-8

# In[53]:


import networkx as nx #networks package we will use for demos
import matplotlib.pyplot as plt #networkx uses matplotlib to generate its plots

import numpy as np #this is the linear algebra package for working with matrices
import pandas as pd #pandas is a dataframe package that is useful for managing network attributes

import math #use this package to take the log of a scalar, use numpy to take the element-wise log of an array
from numpy.linalg import eig
from numpy.linalg import det
from numpy.linalg import inv


# In[54]:


filename = 'projection_edge_list.txt'

#By using with open... the file will automatically close outside the statement, so you don't need to manually close it
with open(filename, 'r') as f: # Open the file with read only ('r') permissions 
    lines = f.read().splitlines()  # Reads in the file line by line, saves each line as an element in a list

#Let's take a look at what the file looks like
print('Here\'s the first few lines of the file:')
for i in range(5):
    print(lines[i]) #print the first 5 lines of the file
print('Each line of the edge list is formated as node1 node2 with the separating character just a space')
    
# Each line of the edge list is formated as node1 node2 with the separating character just a space
# With this format, we can read the edge list directedly using read_edgelist
G = nx.read_edgelist(filename, create_using = nx.Graph, nodetype = int)
    #We specify using create_using what type of graph to make, and nodetype what we should store the nodes as (i.e. int, float, str)

#Let's see how big this network is
print('\nNumber of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())


# In[55]:


G = nx.read_edgelist(filename, create_using = nx.Graph, nodetype = int)

G.katz
#Get the degree sequence as a list
degree_sequence = [G.degree(node) for node in G.nodes()]

#PLot a normalized histogram of the counts in the degree sequence
#density = True normalizes the Y axis to give a probablity density (like in the textbook), and bins is the number of bins

fig, ax = plt.subplots(figsize = (8, 6)) #Create 6 in x 8 in figure
ax.hist(degree_sequence, density = True, bins = 50); #PLot the histogram
ax.set_title('Degree Distribution', fontsize = 20)
ax.set_ylabel(r"Fraction of nodes of degree k ($p_k$)", fontsize=15) #You can add LaTeX to text using $$ and the r in front of the string
ax.set_xlabel("Degree (k)", fontsize=15)


# In[56]:


len(lines)


# In[57]:


matrix = np.zeros((6486,6486))


# In[58]:


for i in range(len(lines)):
    a,b = lines[i].split()
    matrix[int(a)][int(b)] += 1
    matrix[int(b)][int(a)] += 1


# In[59]:


w,v=eig(matrix)
print('E-value:', w)
print('E-vector', v)


# In[63]:


max(w)


# In[64]:


1/max(w)


# In[10]:


I = np.identity(6486)


# In[17]:


a_A = 0.0025 * matrix


# In[18]:


one = np.ones((6486,1))


# In[19]:


x = inv(I - a_A) @ one


# In[20]:


np.shape(x)


# In[80]:





# In[21]:


with open("character_numbers.txt", 'r') as f: # Open the file with read only ('r') permissions 
    characters = f.read().splitlines() 


# In[22]:


dic_char = {}
for i in range(len(x)):
    dic_char[characters[i]] = x[i][0]


# In[23]:


dic_char


# In[31]:



sorted_tuples = sorted(dic_char.items(), key=lambda item: item[1],reverse=True)
print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]
sorted_dict = {k: v for k, v in sorted_tuples}

print(sorted_dict) 


# In[32]:


sorted_dict = {k: v for k, v in sorted_tuples}

print(sorted_dict) 


# In[36]:


with open('Katz_centrality_Marvel.txt', 'w') as f:
    for line in sorted_dict:
        f.write(line)
        f.write("\\ katz centrality: ")
        f.write(str(sorted_dict[line]))
        f.write('\n')


# In[1]:


import pandas as pd
import sklearn.preprocessing

file_name = 'MarvelMovies edgelist.xlsx'

MCU_df = pd.read_excel(file_name, header = None)
le = sklearn.preprocessing.LabelEncoder()
le.fit(list(MCU_df[0]) + list(MCU_df[1]))
list1 = le.transform(MCU_df[0])
list2 = le.transform(MCU_df[1])
edges = list(zip(list1, list2))


# In[52]:


MCU_df[2]


# In[37]:


len(edges) 


# In[9]:


matrix_cinema = np.zeros((19,19))


# In[38]:


for i in range(len(edges)):
    a,b = edges[i]
    matrix_cinema[int(a)][int(b)] += MCU_df[2][i]
    matrix_cinema[int(b)][int(a)] += MCU_df[2][i]


# In[39]:


matrix_cinema


# In[40]:


w_c,v_c=eig(matrix_cinema)
print('E-value:', w_c)
print('E-vector', w_c)


# In[41]:


1/max(w_c)


# In[42]:


I_c = np.identity(19)


# In[43]:


a_A_c = 0.015 * matrix_cinema


# In[44]:


one_c = np.ones((19,1))


# In[45]:


x_c = inv(I_c - a_A_c) @ one_c


# In[46]:


char_name = np.arange(19)


# In[47]:


char_name = le.inverse_transform(char_name)


# In[48]:


dic_c = {}
for i in range(len(char_name)):
    dic_c[char_name[i]] = x_c[i][0]


# In[49]:


dic_c


# In[50]:


sorted_tuples = sorted(dic_c.items(), key=lambda item: item[1],reverse=True)
print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]
sorted_dict_c = {k: v for k, v in sorted_tuples}

print(sorted_dict_c) 


# In[51]:


with open('Katz_centrality_cinema.txt', 'w') as f:
    for line in sorted_dict_c:
        f.write(line)
        f.write("\\ katz centrality: ")
        f.write(str(sorted_dict_c[line]))
        f.write('\n')


# In[ ]:




