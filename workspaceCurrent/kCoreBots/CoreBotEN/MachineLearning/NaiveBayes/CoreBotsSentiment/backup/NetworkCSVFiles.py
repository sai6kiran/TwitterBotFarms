import pandas as pd
import numpy as np
import csv
from os import path
import networkx as nx
import time

'''
def core_number(G):
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

	#print(core)
	return core
find_cores = core_number


def _core_subgraph(G, k_filter, k=None, core=None):
	if core is None:
		core = core_number(G)
	if k is None:
		k = max(core.values())
	nodes = (v for v in core if k_filter(v, k, core))
	return G.subgraph(nodes).copy()


def k_core(G, k=None, core_number=None):
	def k_filter(v, k, c):
		return c[v] >= k
	return _core_subgraph(G, k_filter, k, core_number)

def connected_component_subgraphs(G, copy=True):
	for c in scc(G):
		if copy:
			yield G.subgraph(c).copy()
		else:
			yield G.subgraph(c)


def no_connected_components(G):
	return sum(1 for cc in scc(G))
'''




dfm = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,15,16,19,20], chunksize=2000, skiprows=[0], names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])

dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/ListIDS.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:.0f}'.format
tol = []
uol = []
dfc = pd.DataFrame(columns=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])

#oov = []
pbd = {}
uol = []
'''
for i in dfn.index:
	fsn = "/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Bot-"+str(dfn["userid"][i])+"-EN.csv"
	if(path.isfile(fsn)==True):     #If such a bots csv file exists in the current
		dfi = pd.read_csv(fsn, sep=",", header=None, names=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
		tol += pd.to_numeric(dfi["tweetid"], errors="coerce").astype('int').to_list()
		#uol.append(dfn["userid"][i])
		#for c in range(0, len(dfi["tweetid"].to_list())):
			#pbd[dfi["tweetid"].astype('int64').to_list()[c]] = [dfi["tweetid"].astype('int64').to_list()[c], dfi["userid"].to_list()[c], dfi["tweet_candidate_class"].to_list()[c], dfi["tweet_sentiment_class"].to_list()[c]]
	else:

		print("Broken")
'''
#print(pbd)
uov = []
#uol = list(set(uol))
aov = []
#dfk = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/acp.csv", sep=",", header=None, usecols=[0,1,2,3,4,5], chunksize=2000, names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])
'''
for dft in dfm:
	t0 = time.time()
	dft = dft[dft["in_reply_to_tweetid"].notna()]
	###dft=dft[dft["userid"].isin(uol)]
	###dft = dft[~dft["tweetid"].isin(tol)]
	dft = dft[dft["in_reply_to_tweetid"].astype(int).isin(tol)]
	#dft = dft[~dft["in_reply_to_tweetid"].isin(tol)]
	#uov += dft["tweetid"].to_list()
	#dft[["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"]].to_csv("acc.csv", mode='a', header=False, index=False)
	aov += [[dft["tweetid"].astype('int64').to_list()[k], dft["userid"].to_list()[k], dft["in_reply_to_tweetid"].astype('int64').to_list()[k], dft["in_reply_to_userid"].to_list()[k]] for k in range(0, len(dft["tweetid"].to_list()))]
	#for i in range(0, len(aov)):
		#if("634177729416396800"==str(aov[i][0])):
			#print(aov[i][2])
			#print("break")
			#break
	#uov += dft["userid"].to_list()
	#print(aov)
	#if(not dft.empty):
		#break
	t1 = time.time()
	print(t1-t0)

#pd.DataFrame({"tweetid": uov}).to_csv("temp.csv", mode='a', header=False, index=False)

#uov = list(set(uov))
#print(aov)

print(len(aov))
from collections import OrderedDict
d = OrderedDict()
for t in aov:
	d.setdefault(t[0], t)
aov = list(d.values())
print(len(aov))

ltc = []
ltd = []
#tmp = aov

#ltc = [x+[pbd[x[2]][2]]+[pbd[x[2]][3]] for x in aov if x[1] in uol]

for x in aov:
	if(x[1] in uol):
		print(x)
		x += [pbd[x[2]][2]]+[pbd[x[2]][3]]
		x = x[0:2] + x[4:6]
		ltc.append(x)

	else:
		x += [pbd[x[2]][2]]+[pbd[x[2]][3]]
		x = x[0:2] + x[4:6]
		ltd.append(x)
'''
#dfe = pd.DataFrame(ltc, columns=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
#dfr = pd.DataFrame(ltd, columns=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])


#dfe[["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"]].to_csv("SelfContainedPoliticalNetwork.csv", mode='a', header=False, index=False)
#dfr[["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"]].to_csv("Thirteen.csv", mode='a', header=False, index=False)


dfe = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/SelfContainedPoliticalNetwork.csv", sep=",", header=None, usecols=[0,1,2,3], names=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
dfe.drop_duplicates(subset="tweetid", keep="first")
print(dfe["tweetid"].nunique())
dfe[["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"]].to_csv("SelfContainedPoliticalNetwork.csv", mode='w', header=False, index=False)
'''
dfr = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Thirteen.csv", sep=",", header=None, usecols=[0,1,2,3], names=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
dfr.drop_duplicates(subset="tweetid", keep="first")
print(dfr["tweetid"].nunique())
dfr[["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"]].to_csv("SelfContainedPoliticalNetwork.csv", mode='w', header=False, index=False)
'''

#ltd = [y+[pbd[y[2]][2]]+[pbd[y[2]][3]] for y in aov if y[1] not in uol]


'''
for ubt in aov:
	t0= time.time()
	for tbt in oov:
		if(str(ubt[1]).lower()==str(tbt[1]).lower() and tbt[0]==ubt[3]):
			ltc.append(tbt)
			tmp.remove(ubt)
			break
	t1 = time.time()
	print(t1-t0)

for eac in tmp:
	for ect in oov:
		if(eac[3]==ect[0]):
			ltd.append(ect)
			break

'''
#print(len(ltc))
#print(len(ltd))
#print(ltd)
'''
a = 0
for i in oov:
	if(i not in oov):
		a +=1
		print(i)
print(a)
'''
'''
dfk = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/PoliticaltweetNetwork.csv", sep=",", header=None, usecols=[0,1,2,3,4,5], names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])
print(dfk.count())
dfk.drop_duplicates(subset="tweetid", keep="first")
print(dfk.count())
dfk[["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"]].to_csv("PoliticaltweetNetwork.csv", mode='w', header=False, index=False)
'''
