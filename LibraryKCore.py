#!/usr/bin/env python3
# coding: utf-8

# In[3]:


import csv
import pandas as pd 
from connected_component import connected_component_subgraphs as ccs
from  strongly_connected_component import strongly_connected_components as scc

# In[4]:

'''
df = pd.read_csv("/root/.encrypted/.pythonSai/moreno_highschool/out.moreno_highschool_highschool", sep=" ", header=None, skiprows=2, names=["ndidfr", "ndidto", "weight"]) 
df = df[["ndidfr", "ndidto"]].dropna()

print(df.head())
'''

# ### Undirected Graph:

# In[5]:


import networkx as nx
import matplotlib.pyplot as plt


# In[6]:

'''
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


# plt.savefig("/root/gitlabRepos/python/moreno_highschool/g.pdf", bbox_inches="tight")
plt.show()


# In[7]:

'''
import numpy as np
'''
sni = np.zeros(shape=(70,70), dtype=int)
for index, row in df.iterrows():
    sni[int(row['ndidfr'])-1][int(row['ndidto'])-1] = 1
np.set_printoptions(threshold=np.inf)
tni = sni.transpose()
print(tni)


# In[8]:


nnd = []
for i in range(0, len(tni)):
    tnl = list(tni[i])
    for j in range(0, len(tnl)):
        if(tnl[j]==1):
            nnd.append([i+1, j+1])
# print(nnd)
ndf = pd.DataFrame(data=nnd, columns=['ndideg', 'ndidfr'])
print(ndf)


# In[9]:


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


# plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# ##### KCore of graph:

# In[10]:


for i in range(0, len(G)):
    graphs = ccs(nx.k_core(G,k=i))
    if(graphs is None):
        break
    else:
        for g in graphs:
            print("This is the " + str(i) + " core of graph")
            print(g.edges())
            SCG=nx.Graph()
            scs = []
            for (i,j) in g.edges():
                if(i not in scs):
                    SCG.add_node(i)
                    scs.append(i)
                if(j not in scs):
                    SCG.add_node(j)
                    scs.append(j)

            for (i,j) in g.edges():
                SCG.add_edges_from([(i,j)])

            plt.figure(num=None, figsize=(20, 20), dpi=80)
            plt.axis('off')
            fig = plt.figure(1)

            nx.draw(SCG, with_labels=True, font_size=12)
            plt.show()


# ### Directed Graph:

# In[11]:


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


#plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# In[12]:


import numpy as np
sni = np.zeros(shape=(70,70), dtype=int)
for index, row in df.iterrows():
    sni[int(row['ndidfr'])-1][int(row['ndidto'])-1] = 1
np.set_printoptions(threshold=np.inf)
tni = sni.transpose()
print(tni)


# In[13]:


nnd = []
for i in range(0, len(tni)):
    tnl = list(tni[i])
    for j in range(0, len(tnl)):
        if(tnl[j]==1):
            nnd.append([i+1, j+1])
# print(nnd)
ndf = pd.DataFrame(data=nnd, columns=['ndideg', 'ndidfr'])
print(ndf)


# In[14]:


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


# plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()
'''

# #### KCore of directed graph:

# In[15]:


#    Copyright (C) 2004-2019 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    Antoine Allard <antoine.allard@phy.ulaval.ca>
#    All rights reserved.
#    BSD license.
#
# Authors: Dan Schult (dschult@colgate.edu)
#          Jason Grout (jason-sage@creativetrax.com)
#          Aric Hagberg (hagberg@lanl.gov)
#          Antoine Allard (antoine.allard@phy.ulaval.ca)
"""
Find the k-cores of a graph.

The k-core is found by recursively pruning nodes with degrees less than k.

See the following references for details:

An O(m) Algorithm for Cores Decomposition of Networks
Vladimir Batagelj and Matjaz Zaversnik, 2003.
https://arxiv.org/abs/cs.DS/0310049

Generalized Cores
Vladimir Batagelj and Matjaz Zaversnik, 2002.
https://arxiv.org/pdf/cs/0202039

For directed graphs a more general notion is that of D-cores which
looks at (k, l) restrictions on (in, out) degree. The (k, k) D-core
is the k-core.

D-cores: Measuring Collaboration of Directed Graphs Based on Degeneracy
Christos Giatsidis, Dimitrios M. Thilikos, Michalis Vazirgiannis, ICDM 2011.
http://www.graphdegeneracy.org/dcores_ICDM_2011.pdf

Multi-scale structure and topological anomaly detection via a new network \
statistic: The onion decomposition
L. Hébert-Dufresne, J. A. Grochow, and A. Allard
Scientific Reports 6, 31708 (2016)
http://doi.org/10.1038/srep31708

"""
#import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
from networkx.algorithms.shortest_paths \
    import single_source_shortest_path_length as sp_length

__all__ = ['core_number', 'find_cores', 'k_core', 'k_shell',
           'k_crust', 'k_corona', 'k_truss', 'onion_layers']

def scc_connected_components(G):
	preorder={}
	isconnected = False
	i=0     # Preorder counter
	for source in G:
		if source not in preorder:
			i=i+1
			preorder[source]=i
			source_nbrs=G[source]
			isconnected = False
		else:
			source_nbrs=G[source]
			isconnected = True
		for w in source_nbrs:
			if w not in preorder:
				preorder[w]=i
    
	return i

def strongly_connected_components(G):
    preorder={}
    lowlink={}
    scc_found={}
    scc_queue = []
    i=0     # Preorder counter
    for source in G:
        if source not in scc_found:
            queue=[source]
            while queue:
                v=queue[-1]
                if v not in preorder:
                    i=i+1
                    preorder[v]=i
                done=1
                v_nbrs=G[v]
                for w in v_nbrs:
                    if w not in preorder:
                        queue.append(w)
                        done=0
                        break
                if done==1:
                    lowlink[v]=preorder[v]
                    for w in v_nbrs:
                        if w not in scc_found:
                            if preorder[w]>preorder[v]:
                                lowlink[v]=min([lowlink[v],lowlink[w]])
                            else:
                                lowlink[v]=min([lowlink[v],preorder[w]])
                    queue.pop()
                    if lowlink[v]==preorder[v]:
                        scc_found[v]=True
                        scc=[v]
                        while scc_queue and preorder[scc_queue[-1]]>preorder[v]:
                            k=scc_queue.pop()
                            scc_found[k]=True
                            scc.append(k)
                        yield scc
                    else:
                        scc_queue.append(v)

def strongly_connected_components_recursive(G):
    def visit(v,cnt):
        root[v]=cnt
        visited[v]=cnt
        cnt+=1
        stack.append(v)
        for w in G[v]:
            if w not in visited:
                for c in visit(w,cnt):
                    yield c
            if w not in component:
                root[v]=min(root[v],root[w])
        if root[v]==visited[v]:
            component[v]=root[v]
            tmpc=[v] # hold nodes in this component
            while stack[-1]!=v:
                w=stack.pop()
                component[w]=root[v]
                tmpc.append(w)
            stack.remove(v)
            yield tmpc

    visited={}
    component={}
    root={}
    cnt=0
    stack=[]
    for source in G:
        if source not in visited:
            for c in visit(source,cnt):
                yield c
                
def strongly_connected_component_subgraphs(G, copy=True):
    for comp in strongly_connected_components(G):
        if copy:
            yield G.subgraph(comp).copy()
        else:
            yield G.subgraph(comp)
            
def number_strongly_connected_components(G): 
    return scc_connected_components(G)

def is_strongly_connected(G):
    if len(G)==0:
        raise nx.NetworkXPointlessConcept(
            """Connectivity is undefined for the null graph.""")

    return len(list(strongly_connected_components(G))[0])==len(G)

def core_number(G):
    """Returns the core number for each vertex.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph

    Returns
    -------
    core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXError
        The k-core is not implemented for graphs with self loops
        or parallel edges.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       https://arxiv.org/abs/cs.DS/0310049
    """
    if nx.number_of_selfloops(G) > 0:
        msg = ('Input graph has self loops which is not permitted; '
               'Consider using G.remove_edges_from(nx.selfloop_edges(G)).')
        raise NetworkXError(msg)
    degrees = dict(G.in_degree())
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(G.neighbors(v)) for v in G}
    # print(nbrs)
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                # nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
                
    # print(core)
    return core



find_cores = core_number


def _core_subgraph(G, k_filter, k=None, core=None):
    """Returns the subgraph induced by nodes passing filter `k_filter`.

    Parameters
    ----------
    G : NetworkX graph
       The graph or directed graph to process
    k_filter : filter function
       This function filters the nodes chosen. It takes three inputs:
       A node of G, the filter's cutoff, and the core dict of the graph.
       The function should return a Boolean value.
    k : int, optional
      The order of the core. If not specified use the max core number.
      This value is used as the cutoff for the filter.
    core : dict, optional
      Precomputed core numbers keyed by node for the graph `G`.
      If not specified, the core numbers will be computed from `G`.

    """
    if core is None:
        core = core_number(G)
    if k is None:
        k = max(core.values())
    nodes = (v for v in core if k_filter(v, k, core))
    return G.subgraph(nodes).copy()


def k_core(G, k=None, core_number=None):
    """Returns the k-core of G.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph
    k : int, optional
      The order of the core.  If not specified return the main core.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
      The k-core subgraph

    Raises
    ------
    NetworkXError
      The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    The main core is the core with the largest degree.

    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    See Also
    --------
    core_number

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik,  2003.
       https://arxiv.org/abs/cs.DS/0310049
    """
    def k_filter(v, k, c):
        return c[v] >= k
    return _core_subgraph(G, k_filter, k, core_number)



def k_shell(G, k=None, core_number=None):
    def k_filter(v, k, c):
        return c[v] == k
    return _core_subgraph(G, k_filter, k, core_number)

def connected_component_subgraphs(G, copy=True):
    """Generate connected components as subgraphs.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Returns
    -------
    comp : generator
      A generator of graphs, one for each connected component of G.

    copy: bool (default=True)
      If True make a copy of the graph attributes

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.add_edge(5,6)
    >>> graphs = list(nx.connected_component_subgraphs(G))

    See Also
    --------
    connected_components

    Notes
    -----
    For undirected graphs only.
    Graph, node, and edge attributes are copied to the subgraphs by default.
    """
    for c in scc(G):
        if copy:
            yield G.subgraph(c).copy()
        else:
            yield G.subgraph(c)


def no_connected_components(G):
    """Returns the number of connected components.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Returns
    -------
    n : integer
       Number of connected components

    See Also
    --------
    connected_components
    number_weakly_connected_components
    number_strongly_connected_components

    Notes
    -----
    For undirected graphs only.

    """
    return sum(1 for cc in scc(G))

# In[16]:

'''
for i in range(0, len(G)):
    graphs = connected_component_subgraphs(k_core(G,k=i))
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


# ### Sample Dataset:

# In[17]:


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


# plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()


# In[18]:


for i in range(0, len(G.nodes)-1):
    graphs = list(ccs(nx.k_core(G,k=i)))
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


# In[19]:


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


# plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()
'''


# In[ ]:

'''
for i in range(0, len(G)):
    graphs = connected_component_subgraphs(k_core(G,k=i))
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

'''
# ### Real Life Dataset:

# In[ ]:


#df = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=" ", header=None, skiprows=1, names=["tweetid", "userid"]) 
#df = df[["tweetid", "userid"]].dropna()

#print(df.head())

'''
import csv
from itertools import dropwhile, takewhile

def getstuff(filename, criterion):
	with open(filename, "r") as csvfile:
		datareader = csv.reader(csvfile)
		yield next(datareader)  # yield the header row
		# first row, plus any subsequent rows that match, then stop
		# reading altogether
		# Python 2: use `for row in takewhile(...): yield row` instead
		# instead of `yield from takewhile(...)`.
		print(row[18])
		print(criterion)
		yield from takewhile(
			lambda r: r[18] == criterion,
			dropwhile(lambda r: r[18] != criterion, datareader))
		return

def getdata(filename, criteria):
	for criterion in criteria:
		for row in getstuff(filename, criterion):
			print(row[1])
			yield row
import pdb
pdb.set_trace()
count = 0
for row in getdata("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", "TRUE"):
	count+=1
	#print()
print(count) 
'''

def kcoreDirectedGraph(G, k):
    rrf = False
    U = G.copy()
    gil = list(U.in_degree()) 
    if(k>=0 and type(k)==int):
        for (i,j) in gil:
            if(j<k):
                U.remove_node(i)
                rrf = True
        if(rrf == True):
            return kcoreDirectedGraph(U, k)
        else:
            S = G.copy()
            for i in list(S.nodes()):
                if(not [gni for gni in gil if gni[0]==i]):
                    S.remove_node(i)
            if(S.nodes() is not None):
                return S
            else:
                print("Err")
                return None

def connected_component_subgraphs(G, copy=True):
    for c in scc(G):
        if copy:
            yield G.subgraph(c).copy()
        else:
            yield G.subgraph(c)
'''
import pdb
#pdb.set_trace()
'''
import time
import pdb
#oba = []
rba = []
aih = False
df = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[1,18,19], chunksize=2000, skiprows=1, names=["userid","is_retweet","retweet_userid"])
#df.to_csv('my_parsed.csv', mode='a', header=False)
df_lst = pd.DataFrame(columns=["tweetid","is_retweet", "retweet_userid"])

pd.set_option('display.max_columns', 100)

'''
#Add Retweet Bots to csv file:
for df_ in df:
	#pdb.set_trace()
	t0 = time.time()
	#print("hello")
	#t1 = time.time()
	#print(t1-t0) 
	#pdb.set_trace()
	#print(df_)	
	#break	
	#tmp_df = (df_.rename(columns={col: col.lower() for col in df_.columns}).pipe(lambda x:  x[x.is_retweet == "TRUE"] ))
#for index, row in df.iterrows():
	#if(row["is_retweet"]==True):	
#df_lst = df_lst.append(row, ignore_index=True)
			#print(df_lst)
#print(df_lst)
	df_lst = df_.loc[df_["is_retweet"].map(lambda x: x==True)]
	
	#for index, row in df_lst.iterrows():
	#	if(row["retweet_userid"] not in rba):
	#		rba.append(row["retweet_userid"])
	#		oba.append(row["retweet_userid"])

	#for bui in oba:
		#print(type(bui))
	#	for row in df_.itertuples():
	#		if(row.userid==bui and bui in oba):
	#			oba.remove(bui)
				#df_lst.append(row)
	

#df_lst = df[df.columns[df["is_retweet"].isin([True])]]
#print(df_lst.loc[df_lst["is_retweet"].isin([True])])
	#df_lst.drop(df_lst.columns[[0, 2]], axis=1)
	#if(aih is False):
		#df_lst.to_csv('my_parsed3.csv', mode='a', columns=["tweetid","retweet_userid"], header=True)
		#aih = True
	#else:
	df_lst[["userid","retweet_userid"]].to_csv('my_parsed3.csv', mode='a', header=False, index=False)
	t1 = time.time()
	#print(t1-t0)

'''
'''
def converter(x):
    if isinstance(x, pd.Series):
        return str(x.values)
    else:
        return x
'''
'''
#Add originator bots to csv file:
dfa = pd.read_csv("/root/.encrypted/.pythonSai/my_parsed.csv", sep=",", header=None, usecols=[0,1,2,3,4], chunksize=2000, skiprows=1, names=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])
for dfa in df:
	#t0 = time.time()
	for df_ in df:
		#t3 = time.time()
		df_lst = dfa.loc[dfa["retweet_userid"].map(lambda x: str(x)==df_["userid"].apply(converter).unique().tostring())]
		df_lst.to_csv('my_parsed1.csv', mode='a', header=False)
		#t2 = time.time()
		#print(t2-t3)
	t1 = time.time()
	#print(t1-t0)	
#df_lst.to_csv('my_parsed1.csv', mode='a', header=False)

#for df_ in df:
#	print(type(df_["userid"].apply(converter).unique().tostring()))
'''
'''
for bui in oba:
	for df_ in df:
		dfa.append(df_.loc[df_["userid"].map(lambda x: x==bui, oba.remove(bui))])
		break
'''
'''
dfa = pd.read_csv("/root/.encrypted/.pythonSai/my_parsed.csv", sep=",", header=None, usecols=[0,1,18,19,20], chunksize=2000, skiprows=1, names=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])
for dfn in dfa:
	for df_ in df:
		dfo = dfn.loc[dfn["retweet_userid"].map(lambda x: )]
dfo.to_csv('my_parsed.csv', mode='a', header=False)

'''


#Constructing the graph:
dfn = pd.read_csv("/root/.encrypted/.pythonSai/my_parsed3.csv", sep=",", header=None, chunksize=2000, names=["userid","retweet_userid"])

G=nx.DiGraph()
tmp = 0
cnt = 0

#def addGraph(x):
	#G.add_node(x)

for df_ in dfn:
	#t0 = time.time()
	#tmp +=1
	#print(len(df_))
	#for index, row in df_.iterrows():
		#cnt+=1
		#if(row["retweet_userid"] == ''):
			#print(row, tmp, cnt)
			#break
	#df_lst = df_.loc[df_["retweet_userid"].map(lambda x: print(x, tmp) if x!='' else "false")]
	G.add_edges_from(df_.values)
	#df_lst = df_.loc[df_["userid"].map(lambda x: G.add_node(x))]
	#t1 = time.time()
	#print("************************************************************")
	#print(t1-t0)
	#if(row['retweet_userid'] not in uid and row['retweet_userid'] is not None):
		#G.add_node(row['retweet_userid'])
	#if(row['userid'] not in uid):
		#G.add_node(row['userid'])



'''
'''
'''
for index, row in df_lst.iterrows():
	G.add_edges_from([(row['userid'],row['retweet_userid'])])

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

nx.draw(G, with_labels=True, font_size=12)


plt.savefig("C:\\Users\\user\\Documents\\g.pdf", bbox_inches="tight")
plt.show()
#print(G.nodes, G.edges)

kvl = 0
for i in range(0, len(G)):
	# pdb.set_trace()
'''	



#print("yo")
G.remove_edges_from(nx.selfloop_edges(G))
dkc = []

for i in range(0,120):
	if(i==0):
		t0 = time.time()
		print("hey")
		t1 = time.time()
		print(t1-t0)
		dkc.append((i,len(G.nodes()),number_strongly_connected_components(G)))
		t2 = time.time()
		print(t2-t0)
        
	else:
		t0 = time.time()
		print("hey")
		kcg = k_core(G,k=i)
		t1 = time.time()
		print(t1-t0)
		dkc.append((i,len(kcg.nodes()),number_strongly_connected_components(kcg)))
		t2 = time.time()
		print(t2-t0)
dfn = pd.DataFrame(dkc, columns=["kVal","NoOfCoreBots","NoOfCoreCC"])
dfn.to_csv('kCoreStatGraphCC.csv', mode='a', header=True)


	#print(t1-t0)
#print("The total no. of vertices of graph is: " + str(len(G.nodes())) +  " total no. of edges of graph is: " + str(len(G.edges))+ "\nThe total no. of vertices in core graph is: " + str(len(kcg.nodes())) + " total no. of edges of graph is: " + str(len(kcg.edges)))
#kel = [e for e in kcg.edges]
#dfn = pd.DataFrame(kel, columns=["userid","retweet_userid"])
#dfn[["userid","retweet_userid"]].to_csv('kcoreresults.csv', mode='a', header=False, index=False)

#dfn.to_csv('kcoreresults.csv', mode='a', header=False)
'''	
	if(not kcg.nodes()):
		kvl = i-1
		break
	
if(kvl>0):
	print("Max. K core of graph is: " + str(kvl))
	mkg =kcoreDirectedGraph(G,k=kvl)
	print("The total no. of vertices of graph is: " + str(len(G.nodes())) + "\nThe total no. of vertices in core graph is: " + str(len(mkg.nodes())))
	plt.figure(num=None, figsize=(20, 20), dpi=80)
	plt.axis('off')	
	fig = plt.figure(1)		
	nx.draw(mkg, with_labels=True, font_size=12)
	plt.show()
else:
	print("Max. K core of graph is the: " + str(kvl) + " core.")
#df_final = pd.concat(df_lst)
#print(df_final)
# In[ ]:
'''




