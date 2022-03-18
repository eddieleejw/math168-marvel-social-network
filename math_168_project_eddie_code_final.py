# -*- coding: utf-8 -*-
"""
Math 168 Project Eddie Code Final.ipynb
"""

import pandas as pd
import math
import numpy as np
from scipy.linalg import eigh
import sklearn.preprocessing
import ast

import networkx as nx
from matplotlib import pyplot as plt

"""#General function declaration"""

def spectral_max(M, node_list, M0, edges):
  '''
  Recursive spectral modularity maximization algorithm

  parameters:
  M: current (possible reduced) modularity matrix
  node_list: sequential list of nodes for this current modulairty matrix (starting from 0)
  M0: original modulairty matrix
  edges: number of edges

  return:
  list of groups
  '''


  if len(node_list) == 1: #base case. do nothing
    return [node_list]
  #find lading eigenvector for the modualirty matix
  w,v = eigh(M, eigvals=(M.shape[0] - 1,M.shape[0] - 1))

  #split the nodes into two groups
  group = [[],[]]
  #this will be our vector of s_i (+1 or -1 depending on group)
  s = []
  for i in range(len(v)):
    if v[i] >= 0:
      group[0].append(node_list[i])
      s.append(1)
    else:
      group[1].append(node_list[i])
      s.append(-1)

  #stop here, and check that we actually made a grouping i.e. we didn't assign all the nodes into one group
  #if we did, then stop this recursive call, because this indicates modulairty cannot be increased

  if len(group[0]) == 0 or len(group[1])==0:
    return [node_list]

  s = np.array(s)
  #check if we increased modularity

  Q =  np.matmul(np.matmul(s, M),s)
  Q = (1/(4*edges))*Q


  if Q > 0: #we have increased modulairty, we we recurse
  
    #recurse on each group. before that, we need to define their new modularity matrices

    #recurse on first group
    list1 = spectral_max(new_modularity(M0, group[0]), group[0], M0, edges)

    #recurse on second group
    list2 = spectral_max(new_modularity(M0, group[1]), group[1], M0,edges)

    return list1 + list2
  
  else: #we have not increased modularity. return original node_list
    return [node_list]



def new_modularity(M0, node_list):
  '''
  Define new modulairty matrix given original modularity matrix and the nodes in the group

  parameters:
  M0: original modularity matrix
  node_list: starts indexing from 0
  '''

  #converting node index to numpy index
  node_list = np.array(node_list)

  #slice our original modulairty matrix
  X = M0[node_list,:][:,node_list]
  
  tempsum = sum(X)

  for i in range(len(X)):
    X[i,i] = X[i,i] - tempsum[i]

  return X

"""#Running algorithm on comic network

First create adjacency matrix of bipartite graph (x)
"""

raw_file = "/content/marvel_data_raw"

with open(raw_file, "r") as f:
  x = f.readlines()

x.pop(0)

for i in range(len(x)):
  x[i] = x[i].replace('\t', ' ')
  x[i] = x[i].replace('\n', '')
  x[i] = x[i].split()
  x[i][0] = int(x[i][0])
  x[i][1] = int(x[i][1])

"""Next, create incidence matrix (B)"""

#Our data set has 6486 characters, and 12942 comic book issues. So our incidence matrix should be 6486 x 12942 in size
B = np.zeros((6486,12942))

for pair in x:
  #pair[0] is the character, pair[1] is the comic book.

  #we subtract one from each because numpy indexes from 0
  #we further subtract 6486 from the second entry to account for the fact that the first comic book issue is numbered as node 6487
  B[pair[0]-1, pair[1]-1-6486] = 1

"""Next, create adjacency matrix of one-mode projection (P)"""

#Since we want to project onto the characters:

P = np.matmul(B, np.transpose(B))

#set all diagonal entries to 0
#since we want an unweighted graph, we set all non-zero entries to be 1

for i in range(P.shape[0]):
  for j in range(P.shape[0]):
    if i == j:
      P[i,j] = 0
    elif P[i,j] != 0:
      P[i,j] = 1

"""Finally, create modularity matrix"""

#first find the degree of each node
k = sum(P)

#and the total number of edges
m = sum(k)/2

#Then create the modulairty matrix (M)
M = np.zeros((6486,6486))

for i in range(6486):
  for j in range(6486):
    M[i,j] = P[i,j] - (k[i]*k[j])/(2*m)

"""Apply function to data set"""

node_list = [i for i in range(6486)]
groupings = spectral_max(M, node_list, M, m)

"""#Analysis"""

len(groupings)

"""The algorithm has identified 4 groups."""

max_node = 0
for group in groupings:
  print(len(group))
  if len(group) > max_node:
    max_node = len(group)

"""The groups have sizes 956, 1756, 1717, and 2057 respectively.

### Centrality scores

Look at the nodes with top 3 centrality scores from each community
"""

with open("/content/marvel_eigenvector_centralities.txt","r") as f:
  marvel_dict_txt = f.read()

marvel_dict = ast.literal_eval(marvel_dict_txt)

top_nodes = [[],[],[],[]]
for i in marvel_dict.keys():
  for j in range(4):
    if i in groupings[j] and len(top_nodes[j]) < 3:
      top_nodes[j].append(i)
  
  if len(groupings[0]) == 3 and len(groupings[1]) == 3 and len(groupings[2]) == 3  and len(groupings[3]) == 3:
    break

"""We can see which nodes have the highest centrality scores:"""

top_nodes

"""#Network visualisation

Visualisation on a small subnetwork
"""

node_list_small = groupings[0][0:125]

node_list_small_shifted = np.array(node_list_small)

#isolate the nodes
P_small = P[node_list_small_shifted, :][:, node_list_small_shifted]


#next, I will remove nodes with some degree < k so that we can see only well connected nodes

cut_off_degree = 2

templist = []
for i in range(P_small.shape[0]):
  #print(P_small[i,:].sum())
  if P_small[i,:].sum() <= cut_off_degree:
    templist.append(i)

templist.reverse()

for i in templist:
  P_small = np.delete(P_small, i, 0)
  P_small = np.delete(P_small, i, 1)

n = P_small.shape[0]

#find modulairty matrix and number of edges

#first find the degree of each node
k_small = sum(P_small)

#and the total number of edges
m_small = sum(k_small)/2

#Then create the modulairty matrix (M)
M_small = np.zeros((n,n))

for i in range(n):
  for j in range(n):
    M_small[i,j] = P_small[i,j] - (k_small[i]*k_small[j])/(2*m_small)


#perform spectral modularity maximisation
node_list = [i for i in range(n)]
groupings_small = spectral_max(M_small, node_list, M_small, m_small)

G = nx.from_numpy_array(P_small, create_using=nx.Graph)
fig, ax = plt.subplots(figsize = (12, 12))
nx.draw(G, node_size = 20)
plt.savefig("subgraph_no_community.png")

colors = ["black" for i in range(n)]
color_dict = {0:"red", 1:"yellow", 2:"blue", 3:"magenta", 4:"purple", 5:"green"}

j=0
for i in range(j,j+5):
  for node in groupings_small[i]:
    colors[node] = color_dict[i-j]

pos = nx.spring_layout(G)
fig, ax = plt.subplots(figsize = (8, 8))
nx.draw(G, pos, node_size = 40, node_color = colors)

plt.savefig("subgraph_community.png")

"""###Visualising the cinematic universe"""

file_name = '/content/MarvelMovies edgelist.xlsx'

MCU_df = pd.read_excel(file_name, header = None)
le = sklearn.preprocessing.LabelEncoder()
le.fit(list(MCU_df[0]) + list(MCU_df[1]))
list1 = le.transform(MCU_df[0])
list2 = le.transform(MCU_df[1])
edges = list(zip(list1, list2))


#read in xlsx file as pandas dataframe
MCU_df = pd.read_excel('/content/MarvelMovies edgelist.xlsx', header = None)


# encode each character as a number
le = sklearn.preprocessing.LabelEncoder()
le.fit(list(MCU_df[0]) + list(MCU_df[1]))
list1 = le.transform(MCU_df[0])
list2 = le.transform(MCU_df[1])


#create tuples from lists, so that each tuple is an edge pair
edges = list(zip(list1, list2))


A = np.zeros((19,19))
for edge in edges:
  A[edge[0], edge[1]] = 1
  A[edge[1], edge[0]] = 1

G = nx.from_numpy_array(A, create_using=nx.Graph)
fig, ax = plt.subplots(figsize = (8, 8))
nx.draw(G, node_size = 100)

plt.savefig("MCU_no_community.png")

"""Next, we perform spectral modularity maximization on the above network, and visualise that"""

#find modulairty matrix and number of edges

#first find the degree of each node
k = sum(A)

#and the total number of edges
m = sum(k)/2

#Then create the modulairty matrix (M)
M = np.zeros((19,19))

for i in range(19):
  for j in range(19):
    M[i,j] = A[i,j] - (k[i]*k[j])/(2*m)


node_list = [i for i in range(19)]
groupings = spectral_max(M, node_list, M, m)

# color the nodes

colors = ["black" for i in range(19)]
color_dict = {0:"blue", 1:"yellow", 2:"red", 3: "green", 4:"black"}

for i in range(2):
  for node in groupings[i]:
    colors[node] = color_dict[i]



fig, ax = plt.subplots(figsize = (8, 8))
nx.draw(G, node_size = 100, node_color = colors)

plt.savefig("MCU_community.png")