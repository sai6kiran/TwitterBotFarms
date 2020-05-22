import pandas as pd
import numpy as np
import time
import networkx as nx
import ast
from iteration_utilities import unique_everseen, duplicates

###################DataFrame:#####################################
#Read .CSV File that contains network of a certain policital tweet transmitted between a sequence of bots. Each row of the dataframe contains a unique network a particular political tweet.
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/network.csv", sep=",", header=None, usecols=[0,1], names=["OringatorBotUserID", "DIRECTEDFLOW"])

#List that contains all political network graphs belonging to a certain bot.
png = []

#Lists used for computation of overall network statistics below:
ebs = []	#Contains list of %s of Each Bot's Network similarity in the 2016 elections. Used to compute average below.
mcb = []	#Contains list of most commonly used bots in each originator's Bot's network.
tbs = []	#Contains list of %s of Total/All bot's networks similarity in the 2016 election. I.e How similar where all the political social troll networks, used in the 2016 elections, with each other? Used to compute average below.
mfb = []	#Contains list of most frequently and commonly used bots during the 2016 elections.

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
	mnn = max(len(G), len(H))    #What is the maximum no. of nodes of the two respective graphs. Used to calculate % below.
	pos = (((len(G)+len(H))-(len(R)))/(mnn))*100


	return (R, non, noe, pos)

#This function prints the statistics of each network:
def printGraphNetworkStatistics(lst):
	global ebs
	for i in range(0, len(lst)-1):
		for j in range(i+1, len(lst)):

			###############Calculate the network statistics using "disjoint Union", "Intersection", and "Difference [Complement]" of two graphs:
			rdg = symmetricComplementFunction(lst[i], lst[j])	#The new resultant tuple that contains the Graph and METRICS of all the nodes and edges that lie in Graph i but not in Graph j and vice versa.
			print("The no. of nodes in both Graph " + str(i) +" and Graph " + str(j) + " that do not belong to each other's complement = " +str(rdg[1]))
			print("The no. of edges in both Graph " + str(i) +" and Graph " + str(j) + " that do not belong to each other's complement = " +str(rdg[2]))
			print("The % similarity between the two graphs = " +str(rdg[3]) + "%")
			print("#####################################################################")
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
	tsa = sum(ebs)/(len(ebs))
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
	tsa = sum(tbs)/(len(tbs))

	#Obtain the list of most frequent commonly used social troll bots used in the 2016 elections:
	mfb = list(unique_everseen(duplicates(mfb)))

	#Print Statements:
	print("The average % of similarty of all politically social troll bot networks used in the 2016 elections = " +str(tsa) +"%")
	print("The list of the most commonly used social troll bots used in the 2016 elections: " +str(mfb))
	print("The no. of most commonly used social troll bots used in the 2016 elections: " +str(len(mfb)))


	return True

####################Main Part of Code:#######################################
for ind in dfn.index:
	if(ind == 0):
		cbi = dfn["OringatorBotUserID"][ind]	#Current Bot UserID

		#Initialize the networkx Graph that contains each political network described above.
		G = nx.Graph()

		#Add all edges and nodes into graph:
		G.add_edges_from(ast.literal_eval(dfn["DIRECTEDFLOW"][ind]))
		png.append(G)
	else:
		#If we are looking at all the political networks used by a particular originator bot during the 2016 electoral campaigns:
		if(dfn["OringatorBotUserID"][ind]==cbi):
			#Initialize the networkx Graph that contains each political network described above.
			G = nx.Graph()

			#Add all edges and nodes into graph:
			G.add_edges_from(ast.literal_eval(dfn["DIRECTEDFLOW"][ind]))
			png.append(G)
		#If we come across a new originator bot userid in the dataframe dueing our iteration above, that has its own set of political bot networks:
		else:
			#Print the statistics of the previous set of all political networks belonging to a certain originator bot.
			print("For Twitter bot: " + str(dfn["OringatorBotUserID"][ind-1]))
			print("**********************************")
			printGraphNetworkStatistics(png)
			printEachNetworkStatistic(png)
			print("\n\n\n")

			#Empty the political network graph list.
			png = []

			#Initialize new networkx Graph that contains each political network described above.
			G = nx.Graph()

			#Add all edges and nodes into graph:
			G.add_edges_from(ast.literal_eval(dfn["DIRECTEDFLOW"][ind]))
			png.append(G)

#Print the network statistics for the final network graph in the dataframe:
print("For Twitter bot: " + str(dfn["OringatorBotUserID"][ind-1]))
print("**********************************")
printGraphNetworkStatistics(png)
printEachNetworkStatistic(png)
print("\n\n\n")
print("##########################FINAL STATISTICS OF THE ENTIRE TWITTER SOCIAL BOT NETWORK:###########################################")
printOverallNetworkStatistic()
