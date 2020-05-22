import pandas as pd
import numpy as np
import time
import networkx as nx
import ast
from iteration_utilities import unique_everseen, duplicates
import operator

###################DataFrame:#####################################
#Read .CSV File that contains networks of all KCore bots between a sequence of bots. Each row of the dataframe contains a unique network of a particular K Degenerate Bot.
dfk = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/KDegenerateBotsNetworks.csv", sep=",", skiprows=[0], header=None, usecols=[0,1], names=["KDegenerateBotUserid", "KDegenerateBotNetwork"])
#Read .CSV File that contains userids of all the K Degenerate Bots:
dfu = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/kCoreBotsList.csv", sep="\n", header=None, usecols=[0], names=["userid"])
#Read main dataframe that has all Twitter bots.
#dfm = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,15,16,19,20], names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])

#Lists used for computation of overall network statistics below:

kbs = []	#K Degenerate Bot List. It is the list that stores the UserIds of all K Degenerate bots.
ibs = []	#Pivotal Bot List. I.e. Similar to pbs and lbs
ebs = []	#List that stores average of all similarities with each k core network used for statisical computation below.

#Dictionary that stores occurences of K DEGENERATE BOTS.
kbd = {}	#K Degenerate Bot Dictionary. See below for more information.
ibd = {}	#Pivotal Bot Dictionary. Similar to kbd.

#Initialize the K Degenerate Bot Dictionary.
for i in dfk["KDegenerateBotUserid"].to_list():
	kbs.append(i)
	kbd[i] = 0
kbs = list(set(kbs))

#This function is used to return a list of common node bots between the two graphs.
def intersection(G, H):
	#Instantiate our graph node lists:
	gnl = list(G.nodes)	#"G" node list
	hnl = list(H.nodes)	#"H" node list

	#Obtain our intersection node list
	inl = [nde for nde in gnl if nde in hnl]	#inl = intersection node list.

	#Initialize our empty Graph R:
	R = nx.Graph()

	#Form our intersection graph:
	for i in inl:
		R.add_node(i)

	return R

#This function is used to produce the disjoint union of two graphs of just node bots.
def disjointNodeUnion(G, H):
	#Instantiate our disjoint Graph Set "R":
	R = nx.Graph()
	#R.graph.update(G.graph)
	#R.graph.update(H.graph)

	# add Node Attributes of both G and H into R:
	for n in list(G.nodes):
		R.add_node(n)
	for n in list(H.nodes):
		R.add_node(n)


	return R

#This function is used to provide METRICS of the nodes and edges that lie in G but not in H and vice versa.
def symmetricComplementFunction(G, H):
	#Step 1: Obtain the disjoint union of the two graphs G and H:
	R = disjointNodeUnion(G, H)

	#Step 2: Calculate the no. of nodes in G but not in H and vice versa:
	non = 2*(len(R)) - (len(G)+len(H))	#No. of nodes

	#Step 3: Calculate the % of similarity between two graphs:
	mnn = min(len(G), len(H))    #What is the minimum no. of nodes of the two respective graphs. Used to calculate % below.
	if(mnn>0):
		pos = (((len(G)+len(H))-(len(R)))/(mnn))*100
	else:
		pos = 0.0

	return (R, non, pos)

#This function prints the statistics of each network:
def printGraphNetworkStatistics(lst):
	global ebs
	global mfb

	id1 = lst[0]	#Twitter Bot Id 1
	G = lst[1]	#Graph of Twitter Bot 1
	id2 = lst[2]	#Twitter Bot Id 2
	H = lst[3]	#Graph of Twitter Bot 2

	###############Calculate the network statistics using "disjoint Union", "Intersection", and "Difference [Complement]" of two graphs:
	rdg = symmetricComplementFunction(G, H)	#The new resultant tuple that contains the Graph and METRICS of all the nodes and edges that lie in Graph i but not in Graph j and vice versa.
	#idg = intersection(G, H)
	print("The no. of nodes in this K DEG. BOT's network " + str(id1) +" = " + str(len(list(G.nodes))))
	print("The no. of nodes in this K DEG. BOT's network " + str(id2) +" = " + str(len(list(H.nodes))))
	print("The no. of common nodes in the two networks = " +str(len(list(rdg[0].nodes))-rdg[1]))
	#print("The no. of nodes in both Political Bot's networks that do not belong to each other's complement = " +str(rdg[1]))
	print("The no. of nodes in both Graph " + str(i) +" and Graph " + str(j) + " that do not belong to each other's complement = " +str(rdg[1]))
	print("The % similarity between the two K DEG. BOT networks = " +str(rdg[2]) + "%")
	print("#####################################################################")
	ebs.append(rdg[2])

	###############Calculate the network statistics using Levensthein Edit Distance between two graphs:
	#led = nx.similarity.optimize_graph_edit_distance(lst[i], lst[j], )	#The levenshtein edit distance between two graphs. A Computational metric used to calculate the similiarty between two graphs.
	#mnn = max(len(lst[i]), len(lst[H]))    #What is the maximum no. of nodes of the two respective graphs. Used to calculate % below.
	#print("The levenshtein edit distance between Graph " + str(i) +" and Graph " + str(j) + " = "  +str(led))
	#print("#####################################################################")
	#print("The % similarity between the two graphs = " +str(((mnn-rdg)/mnn)*100))

	#Obtain the list of common bots of all political networks combined:

	'''
	C = nx.Graph()
	if(mfb and ("NO" not in mfb)):
		for i in mfb:
			C.add_node(i)
		D = intersection(C, idg)
		nod = len(list(D.nodes))	#No. of nodes in intersection obtained as "D".
		if(nod>0):
			mfb = list(D.nodes)
		else:
			mfb = ["No"]
	else:
		mfb = list(idg.nodes)
	'''

	return True

#This function prints the overall network statistics of all bots and social troll farms that influenced the 2016 elections:
def printEachNetworkStatistic(lst):
	#Global Variables:
	global ebs
	global mcb
	global tbs
	global mfb

	#Compute the total similarity average of all networks for a particular bot:
	if(len(ebs)>0):
		tsa = sum(ebs)/(len(ebs))
	else:
		tsa = 0.0
	tbs.append(tsa)

	'''
	#Obtain the list of most frequent commonly used bots used by the originator bot:
	C = nx.Graph()
	for i in range(0, len(lst)):
		if(len(C)==0 or i==0):
			C = lst[i]
		else:
			C = intersection(C, lst[i])
		mfb = mfb + list(C.nodes)
	mcb = list(C.nodes)
	'''

	#Print Statements:
	print("The average % of similarty of all Political Twitter BOT networks used by this bot = " +str(tsa) +"%")
	print("The list of the most commonly used bots by this originator bot is: " +str(mcb))
	print("The no. of most commonly used bots by this originator bot is: " +str(len(mcb)))
	print("#####################################################################")

	#Clear Variables:
	ebs = []
	mcb = []


	return True


def printOverallNetworkStatistic():
	#Global Variables:
	global tbs
	global mfb

	#Compute the total similarity average of all social troll bot networks used in the 2016 elections:
	if(len(ebs)>0):
		tsa = sum(ebs)/(len(ebs))
	else:
		tsa = 0.0
	#Obtain the list of most frequent commonly used social troll bots used in the 2016 elections:
	#mfb = list(set(mfb))

	#Print Statements:
	print("The average % of similarty of all the K DEGENERATE BOT networks [combined] = " +str(tsa) +"%")
	#print("The list of the most commonly used bots used by the Political BOTS: " +str(mfb))
	#print("The no. of most commonly used bots used by the Political BOTS: " +str(len(mfb)))


	return True



#The following function will return a dictionary of the no. of occurences of a K Degenerate Bots in all the of the Political Bots networks combined. I.e. the Key = K Degenerate Bot User ID, Value = No. of occurences of Bot in all of the K Degenerate Bots networks [combined].
#Dictionary is returned sorted in descreasing order.
def KBotCount(G):
        #Global Variables:
        global kbd
        global kbs

        for i in list(G.nodes):
                if(i in kbs):
                        kbd[i] += 1


        return True

####################Main Part of Code:#######################################
nob = dfk["KDegenerateBotUserid"].count()	#The total no. of Bots inside dataframe.
for ind in range(0,nob):
	#t0 = time.time()
	#Obtaining the required set of values for Network 1:
	tb1 = dfk["KDegenerateBotUserid"][ind]        #Troll Bot UserID 1
	tn1 = ast.literal_eval(dfk["KDegenerateBotNetwork"][ind])     #Troll Bot Network 1 as a List

	#Creating Graph of Troll Bot 1:
	B1 = nx.DiGraph()
	for i in tn1:
		B1.add_node(i)

	for jnd in range(ind+1, nob):

		#Obtaining the required set of values for Network 1:
		tb2 = dfk["KDegenerateBotUserid"][jnd]	#Troll Bot UserID 2
		tn2 = ast.literal_eval(dfk["KDegenerateBotNetwork"][jnd])	#Troll Bot Network 1 as a List

		#Creating Graph of Troll Bot 2:
		B2 = nx.DiGraph()
		for j in tn2:
			B2.add_node(j)

		printGraphNetworkStatistics([tb1, B1, tb2, B2])
		print("\n")
		break

	#t1 = time.time()
	#print(t1-t0)

#Print the network statistics for the final network graph in the dataframe:
print("\n\n")
print("##########################FINAL STATISTICS OF THE K DEGENERATE TWITTER SOCIAL BOT NETWORK:###########################################")
printOverallNetworkStatistic()
