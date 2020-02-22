import csv 
import pandas as pd
import time
import pdb #Debugger Library

#Load main csv file into a pandas datframe. The dataframe is essentially an array containing sub dataframes. The sub dataframes contain 2000 tweets [rows] each. This is done using "chunksize" = 2000.
df = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,18,19,20], chunksize=2000, skiprows=1, names=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])

#Create a new dataframe that will only contain the tweets [rows] that were retweets from main file. Each tweet [row] has the originator tweet tweet id, originator tweet userid, its tweet tweet id, its tweet userid, and isretweet, that is a boolean [True, False] of whether the tweet was a retweet. 
df_lst = pd.DataFrame(columns=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])
pd.set_option('display.max_columns', 100)

#Iterating through each sub dataframe inside main dataframe.
for df_ in df:
        #pdb.set_trace() #To start debugger
	#Log start time of parsing a sub dataframe of 2000 tweets.
        t0 = time.time()

	#Map all tweets that were retweets to new dataframe.
        df_lst = df_.loc[df_["is_retweet"].map(lambda x: x==True)]["retweet_userid"].map(lambda x: oba.append(x))

	#Write the new dataframe to a csv file.
        df_lst.to_csv('my_parsed.csv', mode='a', header=False)
        
	#Log end time of parsing the sub dataframe of 2000 tweets.
        t1 = time.time()
	#print total time completed to run iteration.
        print(t1-t0)
