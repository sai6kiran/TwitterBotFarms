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
#Dataframe that will store, a dictionary contaning the network of a certain tweet between a sequence of bots, in each cell.
dfd = pd.DataFrame(columns=["OringatorBotUserID", "DIRECTEDFLOW", "Candidate", "Sentiment"])

til=[]  #List that contains the tweet ids for all the bots in the network.
uil=[]	#List that contais userids of all the bots in the network.
ntl=[]	#List that stores the tuples to create the network of the directed flow between a certain tweet between a sequence of bots.

#Iteratring through each bot userid from above [i.e. iterating through every bot's file in present working directory].
for i in dfn.index:
	if(i>887):
		print(i)
		fsn = "/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Bot-"+str(dfn["userid"][i])+"-EN.csv"
		if(path.isfile(fsn)==True):     #If such a bots csv file exists in the current directory

			#Setup the dataframe that reads each bot's tweet csv file.
			dfi = pd.read_csv(fsn, sep=",", header=None, names=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
			#Iterating through each political tweet sent by both during the 2016 elections read from the bot's file present in the current working directory.
			for j in dfi.index:
				print("start new")
				t0 = time.time()
				ais = "[ '"+str(dfi["userid"][j]) + "' ,'" + str(dfi["tweetid"][j]) + "' ,[] ," + "[] ," + "[]]"
				#Run the mainDataframeScript.py from the os.
				tsv = subprocess.Popen(["python3", "mainDataframeScript.py"]+[ais], stdout=PIPE)	#temporary storage variable.
				#Store the output and error of mainDataframeScript.py print statement.
				(out, err) = tsv.communicate()
				#This makes the wait possible
				p_status = tsv.wait()

				out = ast.literal_eval(out.decode("utf-8"))
				tmp = out[0]	#temporary list
				ntl = ntl + tmp
				#ntl = list(dict.fromkeys(ntl))  #Remove all duplicates from ntl list using "Dictionary" data object.
				#ntl = [(rm1, bti) if isinstance(bti, tuple) is False else bti for bti in ntl]

				tmp = out[1]
				uil = uil + tmp
				uil = list(dict.fromkeys(uil))

				tmp = out[2]
				til = til + tmp
				til = list(dict.fromkeys(til))

				t1 = time.time()
				print("Time taken to run  MainDataframe script = " + str(out[3])+"s")
				print(t1-t0)
				#print(uil)
				if(uil and uil!=['']):
					#print("uil")
					#print(uil)
					#print("ntl")
					#print(ntl)
					#print("til")
					#print(til)
					#print("\n")
					for k in range(0,len(uil)):
						#print(k)
						#print(len(uil))
						#print(til[k])
						#print(ntl[k])
						ais = "['"+str(uil[k]) + "','" + "str(til[k])" + "',[] ," + "[] ,"+ "[]]"
						tsv = subprocess.Popen(["python3", "mainDataframeScript.py"]+[ais], stdout=PIPE)        #temporary storage variable.
						#Store the output and error of mainDataframeScript.py print statement.
						(out, err) = tsv.communicate()
						#This makes the wait possible.
						p_status = tsv.wait()

						out = ast.literal_eval(out.decode("utf-8"))
						tmp = out[0]    #temporary list
						ntl = ntl + tmp
						#ntl = list(dict.fromkeys(ntl))  #Remove all duplicates from ntl list using "Dictionary" data object.
						#ntl = [(rm1, bti) if isinstance(bti, tuple) is False else bti for bti in ntl]

						tmp = out[1]
						uil = uil + tmp
						uil = list(dict.fromkeys(uil))

						tmp = out[2]
						til = til + tmp
						til = list(dict.fromkeys(til))

					#Inserting row into dataframe to store the entire network of bots that transmitted a particular tweet starting from the originator bot. In this case, the "first column"=originator bot userid, "second column"=entire network in list format [i.e. ntl]
					dfd.loc[0] = [dfi["userid"][j], ntl, dfi["tweet_candidate_class"][j], dfi["tweet_sentiment_class"][j]]	#Also stores #Candidate and Sentiment Classes the network of bots is targetting.
					dfd[["OringatorBotUserID", "DIRECTEDFLOW", "Candidate", "Sentiment"]].to_csv("networkUserID.csv", mode='a', header=False, index=False)
					dfd.iloc[0:0]
					ntl=[]
					uil=[]
					til=[]
					break
				else:
					dfd.loc[0] = [dfi["userid"][j], ntl, dfi["tweet_candidate_class"][j], dfi["tweet_sentiment_class"][j]]  #Also stores #Candidate and Sentiment Classes the network of bots is targetting.
					dfd[["OringatorBotUserID", "DIRECTEDFLOW", "Candidate", "Sentiment"]].to_csv("networkUserID.csv", mode='a', header=False, index=False)
					dfd.iloc[0:0]
					ntl=[]
					uil=[]
					til=[]
					break


############Writing the dataframe to a CSV file:
#dfd[["OringatorBotUserID", "DIRECTEDFLOW", "Candidate", "Sentiment"]].to_csv("network.csv", mode='a', header=False, index=False)
