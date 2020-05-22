import pandas as pd
import numpy as np
import time
import networkx as nx
import ast
from iteration_utilities import unique_everseen, duplicates
import operator

###################DataFrame:#####################################
#Read .CSV File that contains network of all Policital BOTS transmitted between a sequence of bots. Each row of the dataframe contains a unique network a particular Political Bot.
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/PoliticalBotsNetworks.csv", sep=",", skiprows=[0], header=None, usecols=[0,1], names=["UserId", "Network"])
#Read .CSV File that contains userids of all KCore bots.
dfk = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/kCoreBotsList.csv", sep="\n", header=None, usecols=[0], names=["userid"])
#Read .CSV File that contains userids of all the Pivotal bots:
dfp = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/PivotalBotList.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])

#Read main dataframe that has all Twitter bots.
#dfm = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,15,16,19,20], names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])

#Lists used for computation of overall network statistics below:

ebs = []	#Contains list of %s of Each Bot's Network similarity in the 2016 elections. Used to compute average below.
pbs = []	#Political Bot List. It is the list that stores the UserIds of all political bots.
kbs = []	#K Degenerate Bot List. It is the list that stores the UserIds of all K Degenerate bots.
ibs = []	#Pivotal Bot List. I.e. Similar to pbs and lbs

#Dictionary that stores occurences of each political bot in all the networks of all political bots used for calculation of statisitcs below:
pbd = {}	#Political Bot Dictionary. I.e. Dictionary of all twitter bots that tweeted, replied to, or retweeted political comments that affected the 2016 elections. The key represents the bot's userid. The value represents no. of occurences. See below for more information.
kbd = {}	#K Degenerate Bot Dictionary. Similar to pbd.
ibd = {}	#Pivotal Bot Dictionary. Similar to pbd and kbd.

#Count Variable:
slb = 0	#No. of bots that are only self looping. I.e They are not connected to any other bot in its generic network except for itself.
scb = 0	#No. of self containg bots. I.e. no. of bots that are linked to atleast one Political Bot in its generic network
kdc = 0	#No. of POLITICAL BOTS that have interacted with KCore bots in their network.
ibc = 0	#No. of POLITICAL BOTS that are directly or indirectly connected to the PiVOTAL BOTS in their network.

#Initialize the Political Bot Dictionary.
for i in dfn["UserId"]:
	pbs.append(i)
	pbd[i] = 0
pbs = list(set(pbs))

#Initialize the K Degenerate Bot Dictionary.
for i in dfk["userid"].to_list():
	kbs.append(i)
	kbd[i] = 0
kbs = list(set(kbs))

#Initialize the Pivotal Bot Dictionary.
for i in dfp["userid"].to_list():
	ibs.append(i)
	ibd[i] = 0
ibs = list(set(ibs))

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
	print("The no. of nodes in this Political Bot's network " + str(id1) +" = " + str(len(list(G.nodes))))
	print("The no. of nodes in this Political Bot's network " + str(id2) +" = " + str(len(list(H.nodes))))
	print("The no. of common nodes in the two networks = " +str(len(list(rdg[0].nodes))-rdg[1]))
	#print("The no. of nodes in both Political Bot's networks that do not belong to each other's complement = " +str(rdg[1]))
	print("The no. of edges in both Graph " + str(i) +" and Graph " + str(j) + " that do not belong to each other's complement = " +str(rdg[1]))
	print("The % similarity between the two Political BOT networks = " +str(rdg[2]) + "%")
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

	#Obtain the list of most frequent commonly used bots used by the originator bot:
	C = nx.Graph()
	for i in range(0, len(lst)):
		if(len(C)==0 or i==0):
			C = lst[i]
		else:
			C = intersection(C, lst[i])
		mfb = mfb + list(C.nodes)
	mcb = list(C.nodes)

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
	print("The average % of similarty of all the Political Twitter BOT networks used in the 2016 elections [combined] = " +str(tsa) +"%")
	#print("The list of the most commonly used bots used by the Political BOTS: " +str(mfb))
	#print("The no. of most commonly used bots used by the Political BOTS: " +str(len(mfb)))


	return True


#The following function will update a dictionary of the no. of occurences of a Political Bot in all the of the Political Bots networks combined. I.e. the Key = Political Bot User ID, Value = No. of occurences of Bot in all of the Political Bots networks [combined].
#Dictionary is returned sorted in descreasing order.
def PoliticalBotCount(G):
	#Global Variables:
	global pbd
	global pbs
	global scb
	global slb
	flg = False

	for i in list(G.nodes):
		if(i in pbs):
			pbd[i] += 1
			flg=True

	if(flg==True):
		scb += 1
	else:
		slb += 1

	return True


#The following function will update a dictionary of the no. of occurences of a K Degenerate Bot in all the of the Political Bots networks combined. I.e. the Key = K Degenerate Bot User ID, Value = No. of occurences of Bot in all of the K Degenerate Bots networks [combined].
#Dictionary is returned sorted in descreasing order.
def KBotCount(G):
        #Global Variables:
	global kbd
	global kbs
	global kdc
	flg = False

	for i in list(G.nodes):
		if(i in kbs):
			kbd[i] += 1
			flg = True

	if(flg==True):
		kdc += 1

	return True


#The following function will update a dictionary of the no. of occurences of a Pivotal Bot in all the of the Political Bots networks combined. I.e. the Key = Pivotal Bot User ID, Value = No. of occurences of Bot in all of the K Degenerate Bots networks [combined].
#Dictionary is returned sorted in descreasing order.
def PivotalBotCount(G):
	#Global Variables:
	global ibd
	global ibs
	global ibc
	flg = False

	for i in list(G.nodes):
		if(i in ibs):
			ibd[i] += 1
			flg = True

	if(flg==True):
		ibc += 1

	return True


####################Main Part of Code:#######################################
nob = dfn["UserId"].count()	#The total no. of Bots inside dataframe.
cnt = 0
for ind in range(0,nob-1):
	t0 = time.time()
	#Obtaining the required set of values for Network 1:
	tb1 = dfn["UserId"][ind]        #Troll Bot UserID 1
	tn1 = ast.literal_eval(dfn["Network"][ind])     #Troll Bot Network 1 as a List

	#Creating Graph of Troll Bot 1:
	B1 = nx.Graph()
	for i in tn1:
		B1.add_node(i)

	PoliticalBotCount(B1)
	KBotCount(B1)

	t1 = time.time()
	#print(t1-t0)

	for jnd in range(ind+1, nob):

		#Obtaining the required set of values for Network 1:
		tb2 = dfn["UserId"][jnd]	#Troll Bot UserID 2
		tn2 = ast.literal_eval(dfn["Network"][jnd])	#Troll Bot Network 1 as a List

		#Creating Graph of Troll Bot 2:
		B2 = nx.Graph()
		for j in tn2:
			B2.add_node(j)

		printGraphNetworkStatistics([tb1, B1, tb2, B2])
		break

#Print the network statistics for the final network graph in the dataframe:
print("\n\n\n")
print("##########################FINAL STATISTICS OF THE ENTIRE TWITTER SOCIAL BOT NETWORK:###########################################")
printOverallNetworkStatistic()
#print("##########################DISTRIBUTION OF POLITICAL BOTS NETWORKS:##################################################")
pbd = dict(sorted(pbd.items(), key=operator.itemgetter(1),reverse=True))
#print(pbd)
print("\n\n")
print("##########################DISTRIBUTION OF POLITICAL BOTS IN THE NETWORKS OF POLITICAL BOTS:##################################################")
for x in list(pbd)[0:22]:
	print("The no. of occurences of " + str(x) + " = " + str(pbd[x]))
print(".")
print(".")
print(".")

for x in list(reversed(list(pbd)))[0:22]:
	print("The no. of occurences of " + str(x) + " = " + str(pbd[x]))

print("Total no. of Political Bots = " + str(nob))
print("Total no. of Political Bots that are contained in its connected network component = " + str(scb))
print("Total no. of Political Bots that only loop to iself or do not have any netowrk associated with it = " + str(slb))
print("\n\n")

print("##########################DISTRIBUTION OF K DEGENERATE BOTS IN THE NETWORKS OF POLITICAL BOTS:##################################################")
kbd = dict((k, v) for k, v in kbd.items() if v > 0)
kbd = dict(sorted(kbd.items(), key=operator.itemgetter(1),reverse=True))
for x in list(kbd):
	print("The no. of occurences of K Degenerate BOT '" + str(x) + "' = " + str(kbd[x]))
print("The total no. of K Degenerate BOTS that were found in the UNION of all the Political BOT Networks combined were = " + str(len(kbd)))
print("The total no. of Political BOTS that have the K Degenerate BOTS in their network = " + str(kdc))
print("\n\n")

print("##########################DISTRIBUTION OF PIVOTAL BOTS IN THE NETWORKS OF POLITICAL BOTS:##################################################")
ibd = dict((k, v) for k, v in ibd.items() if v > 0)
ibd = dict(sorted(ibd.items(), key=operator.itemgetter(1),reverse=True))
for x in list(ibd):
	print("The no. of occurences of PIVOTAL BOT '" + str(x) + "' = " + str(ibd[x]))
print("The total no. of Pivotal BOTS that were found in the UNION of all the Political BOT Networks combined were = " + str(len(ibd)))
print("The total no. of Political BOTS that have the PIVOTAL BOTS in their network = " + str(ibc))
