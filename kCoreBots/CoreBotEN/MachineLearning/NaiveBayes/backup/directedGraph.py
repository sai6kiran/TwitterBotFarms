import pandas as pd
import numpy as np
from os import path
import time
import subprocess
from subprocess import PIPE
import ast

###################DataFrames:#####################################
#Read File of all the unique userids of the bots that tweeted messages regarding to the 2016 elections.
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/ListIDS.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])
#Read entire dataframe that has all bots and its outer level retweeted bots.
dfm = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,15,16,19,20], chunksize=2000, names=["tweetid", "userid", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid"])
#Dataframe that will store, a dictionary contaning the network of a certain tweet between a sequence of bots, in each cell.
dfd = pd.DataFrame(columns=["OringatorBotUserID", "DIRECTEDFLOW", "Candidate", "Sentiment"])

fla = False
flb = False

til=[]  #List that contains the tweet ids for all the bots in the network.
uil=[]	#List that contais userids of all the bots in the network.
ntl=[]	#List that stores the tuples to create the network of the directed flow between a certain tweet between a sequence of bots.

#Iteratring through each bot userid from above [i.e. iterating through every bot's file in present working directory].
for i in dfn.index:
	fsn = "/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Bot-"+str(dfn["userid"][i])+"-EN.csv"
	if(path.isfile(fsn)==True):     #If such a bots csv file exists in the current directory

		#Setup the dataframe that reads each bot's tweet csv file.
		dfi = pd.read_csv(fsn, sep=",", header=None, names=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
		#Iterating through each political tweet sent by both during the 2016 elections read from the bot's file present in the current working directory.
		for j in dfi.index:
			t0 = time.time()
			ais = "[ '"+str(dfi["userid"][j]) + "' ,'" + str(dfi["tweetid"][j]) + "' , [" + ",".join(str(ein) for ein in ntl) + "] ," + "[" + ",".join(str(eiu) for eiu in uil) + "] , [" + ",".join(str(eit) for eit in til) + "]]"
			#Run the mainDataframeScript.py from the os.
			tsv = subprocess.Popen(["python3", "mainDataframeScript.py"]+[ais], stdout=PIPE)	#temporary storage variable.
			#Store the output and error of mainDataframeScript.py print statement.
			(out, err) = tsv.communicate()
			#This makes the wait possible
			p_status = tsv.wait()

			out = ast.literal_eval(out.decode("utf-8"))
			ntl = out[0]
			uil = out[1]
			til = out[2]
			t1 = time.time()
			print(t1-t0)

			if(uil):
				for k in range(0,len(uil)):
					ais = "['"+str(uil[k]) + "','" + str(til[k]) + "', [" + ",".join(str(ein) for ein in ntl) + "] ," + "['" + "','".join(str(eiu) for eiu in uil)  + "'] , ['" + ",".join(str(eit) for eit in til) + "']]"
					tsv = subprocess.Popen(["python3", "mainDataframeScript.py"]+[ais], stdout=PIPE)        #temporary storage variable.
					#Store the output and error of mainDataframeScript.py print statement.
					(out, err) = tsv.communicate()
					#This makes the wait possible.
					p_status = tsv.wait()

					out = ast.literal_eval(out.decode("utf-8"))
					ntl = out[0]
					uil = out[1]
					til = out[2]
					fla = True
					print("k" + str(k))
					if(fla==True):
						print("br1")
						break

			#Inserting row into dataframe to store the entire network of bots that transmitted a particular tweet starting from the originator bot. In this case, the "first column"=originator bot userid, "second column"=entire network in list format [i.e. ntl]
			dfd.loc[i+j] = [dfi["userid"][j], ntl, dfi["tweet_candidate_class"][j], dfi["tweet_sentiment_class"][j]]	#Also stores #Candidate and Sentiment Classes the network of bots is targetting.
			ntl=[]
			uil=[]
			til=[]
			print("j" + str(j))
			if(j==0):
				print("br2")
				dfd[["OringatorBotUserID", "DIRECTEDFLOW"]].to_csv("network.csv", mode='a', header=False, index=False)
				flb = True
				break
		if(i==1):
			print("br3")
			break
