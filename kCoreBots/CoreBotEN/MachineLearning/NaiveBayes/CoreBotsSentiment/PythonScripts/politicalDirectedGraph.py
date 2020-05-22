import pandas as pd
import numpy as np
from os import path
import time
import subprocess
from subprocess import PIPE
import ast
import math

###################DataFrames:#####################################
#Read File of all the unique userids of the bots that tweeted messages regarding to the 2016 elections.
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/ListIDS.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])
#Read CSV File that contains all the political tweets sent by the Political Bots that have a directed network associated with it and are self contained within its strongly connected component.
dft = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Thirteen.csv", sep=",", header=None, usecols=[0,1,2,3], names=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
#Dataframe that will store, a dictionary contaning the network of a certain tweet between a sequence of bots, in each cell.
dfd = pd.DataFrame(columns=["PoliticalBotID", "DIRECTEDFLOW", "Candidate", "Sentiment"])
#Read entire dataframe that stores all the Political tweets, retweets and replied to tweets in ONE DATAFRAME!!!
dfm = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/EntirePoliticalTweetNetwork.csv", sep=",", header=None, usecols=[0,1,2,3,4,5], skiprows=[0], names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])
dfm["retweet_tweetid"] = dfm["retweet_tweetid"].astype('Int64')
dfm["in_reply_to_tweetid"] = dfm["in_reply_to_tweetid"].astype('Int64')

til=[]  #List that contains the tweet ids for all the bots in the network.
uil=[]	#List that contais userids of all the bots in the network.
ntl=[]	#List that stores the tuples to create the network of the directed flow between a certain tweet between a sequence of bots.

#Iteratring through each bot userid from above [i.e. iterating through every bot's file in present working directory].
for i in dft.index:
	if(i>-1):
		print("start new")
		t0 = time.time()
		if(math.isnan(dfm[dfm["userid"]==dft["userid"][i]].iloc[0]["retweet_tweetid"])==False and dfm[dfm["userid"]==dft["userid"][i]].iloc[0]["retweet_tweetid"] is not None):
			til += [dft["tweetid"][i], dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["retweet_tweetid"]]
			uil += [dft["userid"][i], dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["retweet_userid"]]
			ntl += [(dft["userid"][i], dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["retweet_userid"])]
			ais = "[ '"+str(dfm[dfm["userid"]==dft["userid"][i]].iloc[0]["retweet_userid"]) + "' ,'" + str(dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["retweet_tweetid"]) + "' ,[] ," + "[] ,['" + "','".join(str(eit) for eit in til) +"']]"
		elif(math.isnan(dfm[dfm["userid"]==dft["userid"][i]].iloc[0]["in_reply_to_tweetid"])==False and dfm[dfm["userid"]==dft["userid"][i]].iloc[0]["in_reply_to_tweetid"] is not None):
			til += [dft["tweetid"][i], dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["in_reply_to_tweetid"]]
			uil += [dft["userid"][i], dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["in_reply_to_userid"]]
			ntl += [(dft["userid"][i], dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["in_reply_to_userid"])]
			ais = "[ '"+str(dfm[dfm["userid"]==dft["userid"][i]].iloc[0]["in_reply_to_userid"]) + "' ,'" + str(dfm[dfm["tweetid"]==dft["tweetid"][i]].iloc[0]["in_reply_to_tweetid"]) + "' ,[] ," + "[] , ['" + "','".join(str(eit) for eit in til) +"']]"
		else:
			til += [dft["tweetid"][i]]
			uil += [dft["userid"][i]]
			ntl += [(dft["tweetid"][i], dft["userid"][i])]
			ais = "[ '"+str(dft["userid"][i]) + "' ,'" + str(dft["tweetid"][i]) + "' ,[] ," + "[] ,['" + "','".join(str(eit) for eit in til) + "']]"

		#Run the mainDataframeScript.py from the os.
		tsv = subprocess.Popen(["python3", "smallerDataFrameScript.py"]+[ais], stdout=PIPE)	#temporary storage variable.
		#Store the output and error of mainDataframeScript.py print statement.
		(out, err) = tsv.communicate()
		#This makes the wait possible
		p_status = tsv.wait()

		out = ast.literal_eval(out.decode("utf-8"))
		tmp = out[0]	#temporary list
		ntl = ntl + tmp
		lon = len(ntl)  #Length of ntl
		ntl = ntl[0:1] + ntl[2:lon]

		tmp = out[1]
		uil = uil + tmp
		uil = list(dict.fromkeys(uil))

		tmp = out[2]
		til = til + tmp
		til = list(dict.fromkeys(til))

		t1 = time.time()
		print("Time taken to run  MainDataframe script = " + str(out[3])+"s")
		print(t1-t0)

		if(til and til!=['']):
			for k in range(0,len(uil)):
				ais = "['"+str(uil[k]) + "','" + "str(til[k])" + "',[] ," + "[] ,['"+ "','".join(str(eit) for eit in til) + "']]"
				tsv = subprocess.Popen(["python3", "smallerDataFrameScript.py"]+[ais], stdout=PIPE)        #temporary storage variable.

				#Store the output and error of mainDataframeScript.py print statement.
				(out, err) = tsv.communicate()
				#This makes the wait possible.
				p_status = tsv.wait()

				out = ast.literal_eval(out.decode("utf-8"))
				tmp = out[0]    #temporary list
				ntl = ntl + tmp

				tmp = out[1]
				uil = uil + tmp
				uil = list(dict.fromkeys(uil))

				tmp = out[2]
				til = til + tmp
				til = list(dict.fromkeys(til))

			#Inserting row into dataframe to store the entire network of bots that transmitted a particular tweet starting from the originator bot. In this case, the "first column"=originator bot userid, "second column"=entire network in list format [i.e. ntl]
			dfd.loc[0] = [dft["tweetid"][i], ntl, dft["tweet_candidate_class"][i], dft["tweet_sentiment_class"][i]]	#Also stores #Candidate and Sentiment Classes the network of bots is targetting.
			dfd[["PoliticalBotID", "DIRECTEDFLOW", "Candidate", "Sentiment"]].to_csv("ThirteenPoliticalTweetNetwork.csv", mode='a', header=False, index=False)
			dfd.iloc[0:0]
			ntl=[]
			uil=[]
			til=[]

		else:
			dfd.loc[0] = [dft["tweetid"][i], ntl, dft["tweet_candidate_class"][i], dft["tweet_sentiment_class"][i]]  #Also stores #Candidate and Sentiment Classes the network of bots is targetting.
			dfd[["PoliticalBotID", "DIRECTEDFLOW", "Candidate", "Sentiment"]].to_csv("ThirteenPoliticalTweetNetwork.csv", mode='a', header=False, index=False)
			dfd.iloc[0:0]
			ntl=[]
			uil=[]
			til=[]


############Writing the dataframe to a CSV file:##############################
dfd[["OringatorBotUserID", "DIRECTEDFLOW", "Candidate", "Sentiment"]].to_csv("network.csv", mode='a', header=False, index=False)
