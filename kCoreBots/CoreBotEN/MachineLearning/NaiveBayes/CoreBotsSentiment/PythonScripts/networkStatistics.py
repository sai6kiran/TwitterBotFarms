import pandas as pd
import numpy as np
import time
import networkx as nx
import ast
from iteration_utilities import unique_everseen, duplicates


#########################DataFrame:######################################################################
#Read .CSV File that contains network of a certain policital tweet transmitted between a sequence of bots. Each row of the dataframe contains a unique network a particular political tweet.
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/PoliticalTweetNetworkModified.csv", sep=",", header=None, usecols=[0,1,2,3], names=["OringatorBotUserID", "DIRECTEDFLOW", "Candidate", "Sentiment"])
#Read main dataframe that has all Twitter bots.
#dfm = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,15,16,19,20], names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])
#Read .CSV File that contains userids of all KCore bots.
dfk = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/kCoreBotsList.csv", sep="\n", header=None, usecols=[0], names=["userid"])
#Read .CSV File that contains userids of all the Pivotal bots:
dfp = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/PivotalBotList.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])
#Read .CSV File that contains userids of all the Political bots:
dfP = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/ListIDS.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])

###########################Statistical Variables:########################################################
#List that contains all political network graphs belonging to a certain bot.
png = []

#Conjoint Network used to unionize all the political networks of all political tweets to compute no. of strongly connected components.
C = nx.DiGraph()

#Lists used for computation of overall network statistics below:
ebs = []	#Contains list of %s of Each Bot's Network similarity in the 2016 elections. Used to compute average below.
mcb = []	#Contains list of most commonly used bots in each originator's Bot's network.
tbs = []	#Contains list of %s of Total/All bot's networks similarity in the 2016 election. I.e How similar where all the political social troll networks, used in the 2016 elections, with each other? Used to compute average below.
mfb = []	#Contains list of most frequently and commonly used bots during the 2016 elections.
apb = []	#List that contains all the political social troll bots in this Twitter network.

#Dictionary that stores lists used to calculate demographic statistics below:
pbd = {}	#Political Bot Dictionary. I.e. Dictionary of all twitter bots that tweeted, replied to, or retweeted political comments that affected the 2016 elections. The key represents the bot's userid. The value is a matrix contaning the bot's candidate class, sentiment class, and type class in each column respetively. See below for more information.


#########################K DEGENERATE AND PIVOTAL BOT REQUIRED VARIABLES:####################################
#Lists:
kbs = []        #K Degenerate Bot List. It is the list that stores the UserIds of all K Degenerate bots.
ibs = []        #Pivotal Bot List. I.e. Similar to pbs and lbs
Pbs = []	#Political Bot List.
ubs = []	#Unique Pivotal Bot List. I.e List that contains Pivotal Bots that are not the K Degenerate nodes or the Political Bot nodes.

#Dictionaries:
kbd = {}        #K Degenerate Bot Dictionary.
ibd = {}        #Pivotal Bot Dictionary.
Pbd = {}	#Political Bot Dictionary.
ubd = {}	#Unique Pivotal Bot Dictionary

#Count Variables:
npk = 0	#Total no. of Political bots that sent out political tweets that reached the K Degenerate bots.
nPk = 0	#Total no. of Political bots that sent out political tweets that reached the Pivotal bots.
npp = 0	#Total no. of Political bots that sent out political tweets that reached the Pivotal bots.
nup = 0	#Total no. of Political bots that sent out political tweets that reached the Unique Pivotal bots.

#Initialize the K Degenerate Bot Dictionary:
for i in dfk["userid"].to_list():
	kbs.append(i)
	kbd[i] = 0
kbs = list(set(kbs))

#Initialize the Pivotal Bot Dictionary.
for i in dfp["userid"].to_list():
	ibs.append(i)
	ibd[i] = 0
ibs = list(set(ibs))

#Initialize the ListID Bot Dictionary:
for i in dfP["userid"].to_list():
	Pbs.append(i)
	Pbd[i] = 0
pbs = list(set(Pbs))

#Initialize the Unique Pivotal Bot Dictionary:
for i in dfp["userid"].to_list():
	if(i not in pbs and i not in kbs):
		ubs.append(i)
		ubd[i] = 0
ubs = list(set(ubs))

############################Information for Political Bot Dictionary defined above:##############################
#Candidate Bot Column. I.e. the candidates each bot, in the political bot dictionary defined above, was targetting in the 2016 elections.
#Sentiment Bot Column. I.e. the sentiments of each bot, in the political bot dictionary defined above, in the 2016 elections.
#Type Bot Column. I.e the types of each bot, in the political bot dictionary defined above, in the 2016 elections. I.e. Is bot ProTrump? AntiHillary?, etc.

#This function returns the type class a certain bot in a political network belongs to:
def typeValue(lst):
	ccv = lst[0]	#Candidate Class Value. I.e. Trump, or Hillary, or Neutral
	scv = lst[1]	#Sentiment Class Value. I.e. positive, negative, or neutral.
	tcv = " "	#Type Class value. I.e. ProTrump, AntiHillary, etc.

	if(ccv.lower()=="trump" and scv.lower()=="positive"):
		tcv = "ProTrump"
	if(ccv.lower()=="trump" and scv.lower()=="negative"):
		tcv = "AntiTrump"
	if(ccv.lower()=="hillary" and scv.lower()=="positive"):
		tcv = "ProHillary"
	if(ccv.lower()=="hillary" and scv.lower()=="negative"):
		tcv = "AntiHillary"
	if(ccv.lower()=="neutral" or scv.lower()=="neutral"):
		tcv = "Neutral"


	return tcv


#This function is used to obtain the Strongly connected component of the union of all the Political networks [combined]:
def scc_connected_components(G):
	preorder={}
	nbl = []        #Node Bot List
	isconnected = False
	i=0     # Preorder counter
	for source in G:
		ise = False     #Boolean: is exists?
		csl = []        #Core Source List
		spn = list(G.predecessors(source))      #Source predecessor node list
		ssn = list(G.successors(source))        #Source successor node list
		for pnb in spn: #pnb = "predecessor node bot"
			if(pnb not in nbl):
				nbl.append(pnb)
				csl.append(pnb)
			else:
				if(pnb not in csl):
					ise = True
		for snb in ssn: #snb = "sucessor node bot"
			if(snb not in nbl):
				nbl.append(snb)
				csl.append(pnb)
			else:
				if(snb not in csl):
					ise = True
		if source not in preorder:
			if(source in nbl or ise==True):
				preorder[source]=i
				source_nbrs=G[source]
				isconnected = False
			else:
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
	return (preorder,i)


#This function is used to return the no. of strongly connected components of the union of all the Political networks [combined]:
def number_strongly_connected_components(G):
	return scc_connected_components(G)[1]


#This function is used to produce the intersection to two graphs.
def intersection(G, H):
	#Instantiate our intersecting graph R as a copy of G:
	R = nx.create_empty_copy(G)

	#Save the edges of G into a temporary variable used below:
	edges = G.edges()

	#Add all the edges present in both G and H into R:
	for e in edges:
		if not H.has_edge(*e):
			R.add_edge(*e)


	return R

#This function is used to produce the disjoint union of two graphs.
def disjointUnion(G, H):
	#Instantiate our disjoint Graph Set "R":
	R = G.__class__()
	R.graph.update(G.graph)
	R.graph.update(H.graph)

	#Save the edges of G and H into temporary variable used below:
	G_edges = G.edges(data=True)
	H_edges = H.edges(data=True)

	# add Nodes and Edges from G into R:
	R.add_nodes_from(G)
	R.add_edges_from(G_edges)
	# add Nodes and Edges from H into R:
	R.add_nodes_from(H)
	R.add_edges_from(H_edges)
	# add Node Attributes of both G and H into R:
	for n in G:
		R.nodes[n].update(G.nodes[n])
	for n in H:
		R.nodes[n].update(H.nodes[n])


	return R

#This function is used to conjointly union all the all the Political networks [combined] used for the calculation of Strongly Connected Component:
def conjointUnion(C, G):
	N = nx.DiGraph
	N = disjointUnion(C, G)
	return N

#This function is used to provide METRICS of the nodes and edges that lie in G but not in H and vice versa.
def symmetricComplementFunction(G, H):
	#Step 1: Obtain the disjoint union of the two graphs G and H:
	R = disjointUnion(G, H)

	#Step 2: Calculate the no. of nodes in G but not in H and vice versa:
	non = 2*(len(R)) - (len(G)+len(H))	#No. of nodes

	#Step 3: Calculate the no. of edges in G but not in H and vice versa:
	I = intersection(G, H) #Graph that is the intersection of graphs G and H
	noe = len(R) - len(I)	#no. of non existant edges in the intersection to two graphs.

	#Step 4: Calculate the % of similarity between two graphs:
	mnn = min(len(G), len(H))    #What is the maximum no. of nodes of the two respective graphs. Used to calculate % below.
	if(mnn>0):
		pos = (((len(G)+len(H))-(len(R)))/(mnn))*100
	else:
		pos = 0.0

	return (R, non, noe, pos)

#The following function will update a dictionary of whether a K Degenerate Bot appeared in all the of the Political Bots political tweet networks combined. I.e. the Key = K Degenerate Bot User ID, Value = {1 if appeared, 0 if not}.
#Dictionary is returned sorted in descreasing order.
def KBotCount(G):
	#Global Variables:
	global kbd
	global kbs
	global npk
	flg = False

	for i in list(G.nodes):
		if(i in kbs):
			kbd[i] = 1
			flg = True

	if(flg==True):
		npk += 1

	return True

#The following function will update a dictionary of whether a Pivotal Bot appeared in all the of the Political Bots political tweet networks combined. I.e. the Key = Pivotal Bot User ID, Value = {1 if appeared, 0 if not.}
#Dictionary is returned sorted in descreasing order.
def PivotalBotCount(G):
	#Global Variables:
	global ibd
	global ibs
	global nPk
	flg = False

	for i in list(G.nodes):
		if(i in ibs):
			ibd[i] = 1
			flg = True

	if(flg==True):
		nPk += 1

	return True

#The following function will update a dictionary of whether a Political Bot appeared in all the of the Political Bots political tweet networks combined. I.e. the Key = Political Bot User ID, Value = {1 if appeared, 0 if not.}
#Dictionary is returned sorted in descreasing order.
def PoliticalBotCount(G):
	#Global Variables:
	global Pbd
	global Pbs
	global npp
	flg = False

	for i in list(G.nodes):
		if(i in Pbs):
			Pbd[i] = 1
			flg = True

	if(flg==True):
		npp += 1

	return True

#The following function will update a dictionary of whether a Unique Pivotal Bot appeared in all the of the Political Bots political tweet networks combined. I.e. the Key = Unique Pivotal Bot User ID, Value = {1 if appeared, 0 if not.}
#Dictionary is returned sorted in descreasing order.
def UniquePivotalBotCount(G):
	#Global Variables:
	global ubd
	global ubs
	global nup
	flg = False

	for i in list(G.nodes):
		if(i in ubs):
			ubd[i] = 1
			flg = True

	if(flg==True):
		nup += 1

	return True

#This function prints the statistics of each network:
def printGraphNetworkStatistics(lst):
	global ebs
	for i in range(0, len(lst)-1):
		for j in range(i+1, len(lst)):

			###############Calculate the network statistics using "disjoint Union", "Intersection", and "Difference [Complement]" of two graphs:
			rdg = symmetricComplementFunction(lst[i], lst[j])	#The new resultant tuple that contains the Graph and METRICS of all the nodes and edges that lie in Graph i but not in Graph j and vice versa.
			print("The no. of nodes in Graph " + str(i) +" = " + str(len(lst[i])))
			print("The no. of nodes in Graph " + str(j) +" = " + str(len(lst[j])))
			print("The no. of common nodes in both Graphs = " + str(((len(lst[i])+len(lst[j]))-(rdg[1]))/2))
			print("The no. of nodes in both Graph " + str(i) +" and Graph " + str(j) + " that do not belong to each other's complement = " +str(rdg[1]))
			print("The no. of edges in both Graph " + str(i) +" and Graph " + str(j) + " that do not belong to each other's complement = " +str(rdg[2]))
			print("The % similarity between the two graphs = " +str(rdg[3]) + "%")
			print("#####################################################################")
			if(rdg[3]>0):
				ebs.append(rdg[3])

			###############Calculate the network statistics using Levensthein Edit Distance between two graphs:
			#led = nx.similarity.optimize_graph_edit_distance(lst[i], lst[j], )	#The levenshtein edit distance between two graphs. A Computational metric used to calculate the similiarty between two graphs.
			#mnn = max(len(lst[i]), len(lst[H]))    #What is the maximum no. of nodes of the two respective graphs. Used to calculate % below.
			#print("The levenshtein edit distance between Graph " + str(i) +" and Graph " + str(j) + " = "  +str(led))
			#print("#####################################################################")
			#print("The % similarity between the two graphs = " +str(((mnn-rdg)/mnn)*100))


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

	if(tsa>0):
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
	print("The average % of similarty of all political twitter networks used by this bot = " +str(tsa) +"%")
	###print("The list of the most commonly used bots by this originator bot is: " +str(mcb))
	###print("The no. of most commonly used bots by this originator bot is: " +str(len(mcb)))
	print("#####################################################################")

	#Clear Variables:
	ebs = []
	mcb = []


	return True

def printOverallNetworkStatistic():
	#Global Variables:
	global tbs
	global mfb
	global C

	#Compute the total similarity average of all social troll bot networks used in the 2016 elections:
	if(len(tbs)>0):
		tsa = sum(tbs)/(len(tbs))
	else:
		tsa = 0.0
	#Obtain the list of most frequent commonly used social troll bots used in the 2016 elections:
	mfb = list(unique_everseen(duplicates(mfb)))

	#Print Statements:
	print("The average % of similarty of all politically social troll bot networks used in the 2016 elections = " +str(tsa) +"%")
	###print("The list of the most commonly used social troll bots used in the 2016 elections: " +str(mfb))
	print("The no. of most commonly used social troll bots used in the 2016 elections: " +str(len(mfb)))

	#Demographic Print Statments:
	nbd = dict((k, v) for k, v in kbd.items() if v > 0)
	print("The total no. of K Degenerate BOTS that received the political tweets = " + str(len(nbd)))
	print("The total no. of Political BOTS that sent its political tweets to the K Degenerate BOTS = " + str(npk))

	dbd = dict((k, v) for k, v in Pbd.items() if v > 0)
	print("The total no. of Other Political Bots that received the political tweets = " + str(len(dbd)))
	print("The total no. of Political BOTS that sent its political tweets to Other Political Bots = " + str(npp))

	Nbd = dict((k, v) for k, v in ibd.items() if v > 0)
	print("The total no. of PIVOTAL BOTS that received the political tweets = " + str(len(Nbd)))
	print("The total no. of Political BOTS that sent its political tweets to the PIVOTAL BOTS = " + str(nPk))

	Dbd = dict((k, v) for k, v in ubd.items() if v > 0)
	print("The total no. of Unique Pivotal Bots that received the political tweets = " + str(len(ubd)))
	print("The total no. of Political BOTS that sent its political tweets to the Unique Pivotal Bots = " + str(nup))

	#Strongly Connected Components Print Statement:
	print("The no. of strongly connected components in the union of all Political tweet networks combined = " + str(number_strongly_connected_components(C)))
	#U = C.to_undirected()
	#print("The no. of connected components in the union of all Political tweet networks combined = " + str(nx.number_connected_components(U)))

	return True


####################Main Part of Code:#######################################
for ind in dfn.index:
	if(ind == 0):
		cbi = dfn["OringatorBotUserID"][ind]	#Current Bot UserID

		#Initialize the networkx Graph that contains each political network described above.
		G = nx.DiGraph()

		#Add all edges and nodes into graph:
		G.add_edges_from(ast.literal_eval(dfn["DIRECTEDFLOW"][ind]))

		#Add Graph, its nodes and its attributes into lists and dictionary respectively:
		png.append(G)
		apb += list(G.nodes)
		apb = list(dict.fromkeys(apb))	#Remove all duplicate nodes [i.e. bot userids] from this list.

		#Call the Count Functions:
		KBotCount(G)
		PivotalBotCount(G)
		PoliticalBotCount(G)
		UniquePivotalBotCount(G)

		#Set the conjointUnion Graph as the First Graph in the first iteration:
		C = G

		tcv = dfn["Candidate"][ind]     #Temporary Candidate Variable.
		tsv = dfn["Sentiment"][ind]     #Temporary Sentiment Variable.
		tmp = [tsv, tcv, typeValue([tcv, tsv])]   #Temporary List.

		for ebn in G.nodes:
			if ebn in pbd.keys():
				if tmp not in pbd[ebn]:
					pbd.update(ebn=pbd[ebn]+[tmp])
			else:
				pbd[ebn] = [tmp]
	else:
		#If we are looking at all the political networks used by a particular originator bot during the 2016 electoral campaigns:
		if(dfn["OringatorBotUserID"][ind]==cbi):
			#Initialize the networkx Graph that contains each political network described above.
			G = nx.DiGraph()

			#Add all edges and nodes into graph:
			G.add_edges_from(ast.literal_eval(dfn["DIRECTEDFLOW"][ind]))

			#Add Graph, its nodes and its attributes into list and dictionary respectively:
			png.append(G)
			apb += list(G.nodes)
			apb = list(dict.fromkeys(apb))  #Remove all duplicate nodes [i.e. bot userids] from this list.

			#Call the Count Functions:
			KBotCount(G)
			PivotalBotCount(G)
			PoliticalBotCount(G)
			UniquePivotalBotCount(G)

			#Call the conjointUnion Function:
			C = conjointUnion(C, G)

			tcv = dfn["Candidate"][ind]	#Temporary Candidate Variable.
			tsv = dfn["Sentiment"][ind]	#Temporary Sentiment Variable.
			tmp = [tsv, tcv, typeValue([tcv, tsv])]	#Temporary List.

			for ebn in G.nodes:
				if ebn in pbd.keys():
					if tmp not in pbd[ebn]:
						pbd.update(ebn=pbd[ebn]+[tmp])
				else:
					pbd[ebn] = [tmp]
		#If we come across a new originator bot userid in the dataframe during our iteration above, that has its own set of political bot networks:
		else:
			#Print the statistics of the previous set of all political networks belonging to a certain originator bot.
			print("For Twitter bot: " + str(dfn["OringatorBotUserID"][ind-1]))
			print("**********************************")
			printGraphNetworkStatistics(png)
			printEachNetworkStatistic(png)
			print("\n")

			#Empty the political network graph list.
			png = []

			#Reinitalize the Current Bot UserID
			cbi = dfn["OringatorBotUserID"][ind]

			#Initialize new networkx Graph that contains each political network described above.
			G = nx.DiGraph()

			#Add all edges and nodes into graph:
			G.add_edges_from(ast.literal_eval(dfn["DIRECTEDFLOW"][ind]))

			#Add Graph, its nodes and its attributes into list and dictionary respectively:
			png.append(G)
			apb += list(G.nodes)
			apb = list(dict.fromkeys(apb))  #Remove all duplicate nodes [i.e. bot userids] from this list.

			#Call the Count Functions:
			KBotCount(G)
			PivotalBotCount(G)
			PoliticalBotCount(G)
			UniquePivotalBotCount(G)

			#Call the conjointUnion Function:
			C = conjointUnion(C, G)

			tcv = dfn["Candidate"][ind]     #Temporary Candidate Variable.
			tsv = dfn["Sentiment"][ind]     #Temporary Sentiment Variable.
			tmp = [tsv, tcv, typeValue([tcv, tsv])]   #Temporary List.

			for ebn in G.nodes:
				if ebn in pbd.keys():
					if tmp not in pbd[ebn]:
						pbd.update(ebn=pbd[ebn]+[tmp])
				else:
					pbd[ebn] = [tmp]

#Print the network statistics for the final network graph in the dataframe:
print("For Twitter bot: " + str(dfn["OringatorBotUserID"][ind-1]))
print("**********************************")
printGraphNetworkStatistics(png)
printEachNetworkStatistic(png)
print("\n\n")
print("##########################FINAL STATISTICS OF THE ENTIRE TWITTER SOCIAL BOT NETWORK:###########################################")
printOverallNetworkStatistic()
print("\n\n")

print("##########################DEMOGRAPHICS OF THE ENTIRE TWITTER SOCIAL BOT NETWORK:###############################################")
nob = 205431 #No. of Bots.
nop = 0	#No. of positive sentiment bots.
non = 0	#No. of negative sentiment bots.
noN = 0	#No. of Neutral bots.
NoT = 0	#No. of bots targetting Trump.
NoC = 0	#No. of bots targetting Hillary.
NPT = 0	#No. of PRO TRUMP bots.
NAT = 0	#No. of ANTI TRUMP bots.
NPC = 0	#No. of PRO CLINTON bots.
NAC = 0	#No. of ANTI CLINTON bots.

for all in pbd.values():
	for ebl in all:
		if(ebl[0].lower()=='positive'):
			nop += 1
		if(ebl[0].lower()=='negative'):
			non += 1
		if(ebl[0].lower()=='neutral'):
			noN += 1
		if(ebl[1].lower()=='trump'):
			NoT += 1
		if(ebl[1].lower()=='hillary'):
			NoC += 1
		if(ebl[1].lower()=='neutral'):
			noN += 1
		if(ebl[2].lower()=='protrump'):
			NPT += 1
		if(ebl[2].lower()=='antitrump'):
			NAT += 1
		if(ebl[2].lower()=='prohillary'):
			NPC += 1
		if(ebl[2].lower()=='antihillary'):
			NAC += 1

print("Total no. of bots in the entire twitter network = " + str(nob))
print("Total no. of political social troll bots in this twitter network = " + str(len(apb)))
print("************************************************")
print("Total no. of positive social troll bots in the entire twitter network = " +str(nop))
print("% of positive social troll bots in the 2016 political network = " +str((nop/len(apb))*100) +"%")
print("% of positive social troll bots in the entire twitter network = " +str((nop/nob)*100) +"%")
print("************************************************")
print("Total no. of negative social troll bots in the entire twitter network = " +str(non))
print("% of negative social troll bots in the 2016 political network = " +str((non/len(apb))*100) +"%")
print("% of negative social troll bots in the entire twitter network = " +str((non/nob)*100) +"%")
print("************************************************")
print("Total no. of neutral social troll bots in the entire twitter network = " +str(noN))
print("% of neutral social troll bots in the 2016 political network = " +str((noN/len(apb))*100) +"%")
print("% of neutral social troll bots in the entire twitter network = " +str((noN/nob)*100) +"%")
print("************************************************")
print("Total no. of DONALD TRUMP social troll bots in the entire twitter network = " +str(NoT))
print("% of DONALD TRUMP social troll bots in the 2016 political network = " +str((NoT/len(apb))*100) +"%")
print("% of DONALD TRUMP social troll bots in the entire twitter network = " +str((NoT/nob)*100) +"%")
print("************************************************")
print("Total no. of HILLARY CLINTON social troll bots in the entire twitter network = " +str(NoC))
print("% of HILLARY CLINTON social troll bots in the 2016 political network = " +str((NoC/len(apb))*100) +"%")
print("% of HILLARY CLINTON social troll bots in the entire twitter network = " +str((NoC/nob)*100) +"%")
print("************************************************")
print("Total no. of PRO TRUMP social troll bots in the entire twitter network = " +str(NPT))
print("% of PRO TRUMP social troll bots in the 2016 political network = " +str((NPT/len(apb))*100) +"%")
print("% of PRO TRUMP social troll bots in the entire twitter network = " +str((NPT/nob)*100) +"%")
print("************************************************")
print("Total no. of ANTI TRUMP social troll bots in the entire twitter network = " +str(NAT))
print("% of ANTI TRUMP social troll bots in the 2016 political network = " +str((NAT/len(apb))*100) +"%")
print("% of ANTI TRUMP social troll bots in the entire twitter network = " +str((NAT/nob)*100) +"%")
print("************************************************")
print("Total no. of PRO CLINTON social troll bots in the entire twitter network = " +str(NPC))
print("% of PRO CLINTON social troll bots in the 2016 political network = " +str((NPC/len(apb))*100) +"%")
print("% of PRO CLINTON social troll bots in the entire twitter network = " +str((NPC/nob)*100) +"%")
print("************************************************")
print("Total no. of ANTI CLINTON social troll bots in the entire twitter network = " +str(NAC))
print("% of ANTI CLINTON social troll bots in the 2016 political network = " +str((NAC/len(apb))*100) +"%")
print("% of ANTI CLINTON social troll bots in the entire twitter network = " +str((NAC/nob)*100) +"%")
