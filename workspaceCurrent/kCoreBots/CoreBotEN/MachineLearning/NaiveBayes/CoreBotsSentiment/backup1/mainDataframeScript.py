import pandas as pd
import numpy as np
from os import path
import time
import sys
import recursiveMatchScript as rms
import ast

###################DataFrame:#####################################
#Read entire dataframe that has all bots and its outer level retweeted bots.
dfm = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/tweetRetweet.csv", sep=",", header=None, usecols=[0,1,2,3,4,5], skiprows=[0], chunksize=2000, names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])

###################Function:#####################################
#Reason for this script is because the directedGraph.py that runs below function only runs the inner for loop ["for df_ in dfm:"] once and then results in a StopIteration exception that I could not handle inside the DirectedGraph.py script. Thus, every single time the directedGraph.py needs to use this function, it will run this script as a whole new function.
loa = ast.literal_eval(sys.argv[1])	#List of arguments.
rm1 = str(loa[0])
rm2 = str(loa[1])
ntl = loa[2]
uil = loa[3]
til = loa[4]

#def mainDataframeLoop(rm1, rm2, ntl, idx, uil, til):      #rm1: RecursiveMatch argument 1;        rm2: RecursiveMatch argument 2
t0 = time.time()
for df_ in dfm:
	t0 = time.time()
	tmp = []
	tsv = rms.recursiveMatchFunction(df_, rm1, rm2, uil, til)	#Temporary Storage variable.
	tmp = tsv[0]
	uil = tsv[1]
	til = tsv[2]

	if(tmp):
		ntl = ntl+tmp
		ntl = list(dict.fromkeys(ntl))  #Remove all duplicates from ntl list using "Dictionary" data object.
		ntl = [(rm1, bti) if isinstance(bti, tuple) is False else bti for bti in ntl]
t1 = time.time()

tls = "[["+ ",".join(str(ein) for ein in ntl) +"]" + ", ['" + "','".join(str(eiu) for eiu in uil) +"']" + ", ['" + "','".join(str(eit) for eit in til) +"'] ," + str(t1-t0)+"]"	#Temporary Storage List.
print(tls)
