#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd


# In[2]:


df = pd.read_csv("C:\\Users\\user\\Downloads\\moreno_highschool\\out.moreno_highschool_highschool", sep=" ", header=None, skiprows=2, names=["ndidfr", "ndidto", "weight"]) 
df = df[["ndidfr", "ndidto"]].dropna()

print(df.head())


# ### Undirected Graph:

# In[3]:


import networkx as nx
import matplotlib.pyplot as plt


# In[4]:


G=nx.Graph()
sdt = []
for index, row in df.iterrows():
    if(row['ndidfr'] not in sdt):
        G.add_node(row['ndidfr'])
    if(row['ndidto'] not in sdt):
        G.add_node(row['ndidto'])

for index, row in df.iterrows():
    G.add_edges_from([(row['ndidfr'],row['ndidto'])])

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

nx.draw(G, with_labels=True, font_size=12)


plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# In[5]:


import numpy as np
sni = np.zeros(shape=(70,70), dtype=int)
for index, row in df.iterrows():
    sni[int(row['ndidfr'])-1][int(row['ndidto'])-1] = 1
np.set_printoptions(threshold=np.inf)
tni = sni.transpose()
print(tni)


# In[6]:


nnd = []
for i in range(0, len(tni)):
    tnl = list(tni[i])
    for j in range(0, len(tnl)):
        if(tnl[j]==1):
            nnd.append([i+1, j+1])
# print(nnd)
ndf = pd.DataFrame(data=nnd, columns=['ndideg', 'ndidfr'])
print(ndf)


# In[7]:


G=nx.Graph()
sdt = []
for index, row in ndf.iterrows():
    if(row['ndideg'] not in sdt):
        G.add_node(row['ndideg'])
    if(row['ndidfr'] not in sdt):
        G.add_node(row['ndidfr'])

for index, row in ndf.iterrows():
    G.add_edges_from([(row['ndidfr'],row['ndideg'])])

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

nx.draw(G, with_labels=True, font_size=12)


plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# ##### KCore of graph:

# In[8]:


def kcore(k, G):
    rrf = False
    U = G.copy()
    gil = list(U.degree(list(U))) 
    if(k>=0 and type(k)==int):
        for (i,j) in gil:
            if(j<k):
                U.remove_node(i)
                rrf = True
        if(rrf == True):
            return kcore(k, U)
        else:
            S = G.copy()
            for i in list(S.nodes()):
                if(not [gni for gni in gil if gni[0]==i]):
                    S.remove_node(i)
            # print(S)
            if(S.edges is not None):
                # print(S.edges)
                return list(S.edges)
            else:
                print("Err")
                return None


# In[9]:


for i in range(0, len(G)):
    graphs = kcore(i, G)
    if(not graphs):
        break
    else:
        count = 0
        print("This is the " + str(i) + " core of graph")
        SCG=nx.Graph()
        scs = []
        for (i,j) in graphs:
            if(i not in scs):
                count+=1
                SCG.add_node(i)
                scs.append(i)
            if(j not in scs):
                count+=1
                SCG.add_node(j)
                scs.append(j)
        
        print("The total no. of vertices of graph is: " + str(len(G.nodes())) + ". \nThe total no. of vertices in core graph is:" + str(count))

        for (i,j) in graphs:
            SCG.add_edges_from([(i,j)])

        plt.figure(num=None, figsize=(20, 20), dpi=80)
        plt.axis('off')
        fig = plt.figure(1)

        nx.draw(SCG, with_labels=True, font_size=12)
        plt.show()


# ### Directed Graph:

# In[10]:


G=nx.DiGraph()
sdt = []
for index, row in df.iterrows():
    if(row['ndidfr'] not in sdt):
        G.add_node(row['ndidfr'])
    if(row['ndidto'] not in sdt):
        G.add_node(row['ndidto'])

for index, row in df.iterrows():
    G.add_edges_from([(row['ndidfr'],row['ndidto'])])

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

nx.draw(G, with_labels=True, font_size=12)


plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# In[11]:


import numpy as np
sni = np.zeros(shape=(70,70), dtype=int)
for index, row in df.iterrows():
    sni[int(row['ndidfr'])-1][int(row['ndidto'])-1] = 1
np.set_printoptions(threshold=np.inf)
tni = sni.transpose()
print(tni)


# In[12]:


nnd = []
for i in range(0, len(tni)):
    tnl = list(tni[i])
    for j in range(0, len(tnl)):
        if(tnl[j]==1):
            nnd.append([i+1, j+1])
# print(nnd)
ndf = pd.DataFrame(data=nnd, columns=['ndideg', 'ndidfr'])
print(ndf)


# In[13]:


G=nx.DiGraph()
sdt = []
for index, row in ndf.iterrows():
    if(row['ndideg'] not in sdt):
        G.add_node(row['ndideg'])
    if(row['ndidfr'] not in sdt):
        G.add_node(row['ndidfr'])

for index, row in ndf.iterrows():
    G.add_edges_from([(row['ndidfr'],row['ndideg'])])

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

nx.draw(G, with_labels=True, font_size=12)


plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# In[14]:


def kcoreDirectedGraph(k, G):
    rrf = False
    U = G.copy()
    gil = list(U.in_degree()) 
    if(k>=0 and type(k)==int):
        for (i,j) in gil:
            if(j<k):
                U.remove_node(i)
                rrf = True
        if(rrf == True):
            return kcoreDirectedGraph(k, U)
        else:
            S = G.copy()
            for i in list(S.nodes()):
                if(not [gni for gni in gil if gni[0]==i]):
                    S.remove_node(i)
            # print(S)
            if(S.out_edges is not None):
                # print(S.out_edges)
                return list(S.out_edges)
            else:
                print("Err")
                return None

kcoreDirectedGraph(3,G)


# In[15]:


for i in range(0, len(G)):
    graphs = kcoreDirectedGraph(i, G)
    print(graphs)
    if(not graphs):
        break
    else:
        count = 0
        print("This is the " + str(i) + " core of graph")
        SCG=nx.DiGraph()
        scs = []
        for (i,j) in graphs:
            if(i not in scs):
                count+=1
                SCG.add_node(i)
                scs.append(i)
            if(j not in scs):
                count+=1
                SCG.add_node(j)
                scs.append(j)
        
        print("The total no. of vertices of graph is: " + str(len(G.nodes())) + ". \nThe total no. of vertices in core graph is:" + str(count))

        for (i,j) in graphs:
            SCG.add_edges_from([(i,j)])

        plt.figure(num=None, figsize=(20, 20), dpi=80)
        plt.axis('off')
        fig = plt.figure(1)

        nx.draw(SCG, with_labels=True, font_size=12)
        plt.show()


# ### Sample Dataset:

# In[16]:


G=nx.Graph()
for i in range(1, 22):
    G.add_node(i)


G.add_edges_from([(2,3)])
G.add_edges_from([(3,4)])
G.add_edges_from([(4,5)])
G.add_edges_from([(5,6)])
G.add_edges_from([(6,3)])
G.add_edges_from([(7,8)])
G.add_edges_from([(8,9)])
G.add_edges_from([(8,10)])
G.add_edges_from([(8,11)])
G.add_edges_from([(9,10)])
G.add_edges_from([(10,11)])
G.add_edges_from([(10,12)])
G.add_edges_from([(10,13)])
G.add_edges_from([(10,14)])
G.add_edges_from([(12,13)])
G.add_edges_from([(13,14)])
G.add_edges_from([(12,14)])
G.add_edges_from([(11,15)])
G.add_edges_from([(11,16)])
G.add_edges_from([(11,17)])
G.add_edges_from([(15,16)])
G.add_edges_from([(16,17)])
G.add_edges_from([(15,17)])
G.add_edges_from([(14,18)])
G.add_edges_from([(17,18)])
G.add_edges_from([(14,19)])
G.add_edges_from([(18,19)])
G.add_edges_from([(19,20)])
G.add_edges_from([(19,21)])

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

nx.draw(G, with_labels=True, font_size=12)


plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# In[17]:


for i in range(0, len(G.nodes)-1):
    print(kcore(G,0))
    graphs = list(nx.connected_component_subgraphs(kcore(G,k=i)))
    if(graphs is None):
        break
    else:
        for g in graphs:
            print("This is the " + str(i) + " core of graph")
            print(g.edges())
            SCG=nx.Graph()
            scs = []
            for (r,s) in g.edges():
                if(r not in scs):
                    SCG.add_node(r)
                    scs.append(r)
                if(s not in scs):
                    SCG.add_node(s)
                    scs.append(s)

            for (u,v) in g.edges():
                SCG.add_edges_from([(u,v)])

            plt.figure(num=None, figsize=(20, 20), dpi=80)
            plt.axis('off')
            fig = plt.figure(1)

            nx.draw(SCG, with_labels=True, font_size=12)
            plt.show()


# In[ ]:


G=nx.DiGraph()
for i in range(1, 22):
    G.add_node(i)


G.add_edges_from([(2,3)])
G.add_edges_from([(3,4)])
G.add_edges_from([(4,5)])
G.add_edges_from([(5,6)])
G.add_edges_from([(6,3)])
G.add_edges_from([(7,8)])
G.add_edges_from([(8,9)])
G.add_edges_from([(8,10)])
G.add_edges_from([(8,11)])
G.add_edges_from([(9,10)])
G.add_edges_from([(10,11)])
G.add_edges_from([(10,12)])
G.add_edges_from([(10,13)])
G.add_edges_from([(10,14)])
G.add_edges_from([(12,13)])
G.add_edges_from([(13,14)])
G.add_edges_from([(12,14)])
G.add_edges_from([(11,15)])
G.add_edges_from([(11,16)])
G.add_edges_from([(11,17)])
G.add_edges_from([(15,16)])
G.add_edges_from([(16,17)])
G.add_edges_from([(15,17)])
G.add_edges_from([(14,18)])
G.add_edges_from([(17,18)])
G.add_edges_from([(14,19)])
G.add_edges_from([(18,19)])
G.add_edges_from([(19,20)])
G.add_edges_from([(19,21)])

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

nx.draw(G, with_labels=True, font_size=12)


plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# In[ ]:


for i in range(0, len(G)):
    print(kcoreDirectedGraph(G,k=i))
    graphs = nx.connected_component_subgraphs(kcoreDirectedGraph(G,k=i))
    if(graphs is None):
        break
    else:
        for g in graphs:
            if(g.edges):
                print("This is the " + str(i) + " core of graph")
                # print(g.edges())
                SCG=nx.DiGraph()
                scs = []
                for (r,s) in g.edges():
                    if(r not in scs):
                        SCG.add_node(r)
                        scs.append(r)
                    if(s not in scs):
                        SCG.add_node(s)
                        scs.append(s)

                for (u,v) in g.edges():
                    SCG.add_edges_from([(u,v)])
                print("The total no. of vertices of graph is: " + str(len(G.nodes())) + ". \nThe total no. of vertices in core graph is:" + str(len(g.nodes())))

                plt.figure(num=None, figsize=(20, 20), dpi=80)
                plt.axis('off')
                fig = plt.figure(1)

                nx.draw(SCG, with_labels=True, font_size=12)
                plt.show()


# ### Twitter API:

# In[23]:


get_ipython().system('pip show pandas')


# In[ ]:




