import pandas as pd
import numpy as np
from os import path
import time

#The function recursively matches and returns all the bots that satisfy the conditions that one bot retweeted or replied to the original bot's message.
def recursiveMatchFunction(df_, uid, tid, uil, til):
	t1 = time.time()
	tmd = df_[["tweetid"]].loc[df_["in_reply_to_userid"].map(lambda x: x=="d73d76d5d4b474ab742c52f16b438e7cc29830aa03b9278310151c366efc4773")].iloc[:, 0].tolist()  #List that stores the tweetids of all bots that matches the criterion for in_reply_to_tweetid.
	tmd += df_[["tweetid"]].loc[df_["retweet_userid"].map(lambda x: x=="d73d76d5d4b474ab742c52f16b438e7cc29830aa03b9278310151c366efc4773")].iloc[:, 0].tolist()     #List that stores the tweetids of all bots that matches the criterion for retweet_tweetid.
	umd = df_[["userid"]].loc[df_["in_reply_to_userid"].map(lambda x: x=="d73d76d5d4b474ab742c52f16b438e7cc29830aa03b9278310151c366efc4773")].iloc[:, 0].tolist()   #List that stores the userids of all bots that matches the criterion for in_reply_to_userid.
	umd += df_[["userid"]].loc[df_["retweet_userid"].map(lambda x: x=="d73d76d5d4b474ab742c52f16b438e7cc29830aa03b9278310151c366efc4773")].iloc[:, 0].tolist()      #List that stores the userids of all bots that matches the criterion for retweet_userid.
	'''
	if(tmd):
		dft.loc[idx] = tmd
		idx += len(tmd)
	dft.drop_duplicates()
	'''
	#Store the tweet ids into a list:
	til = til+tmd   #Append all the values from tmd list into loc list
	til = list(dict.fromkeys(til))  #Remove all duplicates from loc list using "Dictionary" data object.
	#Store the user ids into a list:
	uil = uil+umd
	uil = list(dict.fromkeys(uil))

	t2 = time.time()
	#print(t2-t1)
	#print(pd.DataFrame(df_.loc[df_["in_reply_to_userid"].map(lambda x: x=="2518710111")]["userid"]))
	#print(tmd, tid)
	return (umd, uil, til)
