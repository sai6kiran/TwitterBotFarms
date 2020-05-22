import pandas as pd
import numpy as np
from os import path
import time

#The function recursively matches and returns all the bots that satisfy the conditions that one bot retweeted or replied to the original bot's message.
def recursiveMatchFunction(df_, uid, tid, uil, til):
	tmd = df_[["tweetid"]].loc[df_["in_reply_to_tweetid"].map(lambda x: str(x)==tid)].iloc[:, 0].tolist()  #List that stores the tweetids of all bots that matches the criterion for in_reply_to_tweetid.
	tmd += df_[["tweetid"]].loc[df_["retweet_tweetid"].map(lambda x: str(x)==tid)].iloc[:, 0].tolist()     #List that stores the tweetids of all bots that matches the criterion for retweet_tweetid.
	umd = df_[["userid"]].loc[df_["in_reply_to_tweetid"].map(lambda x: str(x)==tid)].iloc[:, 0].tolist()   #List that stores the userids of all bots that matches the criterion for in_reply_to_userid.
	umd += df_[["userid"]].loc[df_["retweet_tweetid"].map(lambda x: str(x)==tid)].iloc[:, 0].tolist()      #List that stores the userids of all bots that matches the criterion for retweet_userid.

	#Store the tweet ids into a list:
	for i in range(0,len(tmd)):
		if(str(tmd[i]) not in til):
			til.append(str(tmd[i]))	#Append all the unique values from tmd list into til list

	#Store the user ids into a list:
	for i in range(0,len(tmd)):
		if(str(tmd[i]) not in til):
			uil.append(str(umd[i]))	#Append all the unique values from umd list into uil list


	return (umd, uil, til)
