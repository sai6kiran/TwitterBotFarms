import csv
import pandas as pd
import time
import pdb #Debugger Library

#Load csv file containing only retweets into a pandas datframe. The dataframe is essentially an array containing sub dataframes. The sub dataframes contain 2000 tweets [rows] each. This is done using "chunksize" = 2000.


dfa = pd.read_csv("/root/.encrypted/.pythonSai/my_parsed.csv", sep=",", header=None, usecols=[0,1,2,3,4], chunksize=2000, skiprows=1, names=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])

#Load main csv file into a pandas datframe. The dataframe is essentially an array containing sub dataframes. The sub dataframes contain 2000 tweets [rows] each. This is done using "chunksize" = 2000.
df = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,18,19,20], chunksize=2000, skiprows=1, names=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])

#Create a new dataframe that will only contain the tweets [rows] that were retweets from main file. Each tweet [row] has the originator tweet tweet id, originator tweet userid, its tweet userid, and isretweet, that is a boolean [True, False] of whether the tweet was a retweet.
df_lst = pd.DataFrame(columns=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])
pd.set_option('display.max_columns', 100)

#Iterating through each sub dataframe inside retweet dataframe.
for dfa in df:
		#pdb.set_trace() #To start debugger
    	#Log start time of parsing a sub dataframe of 2000 tweets.
        t0 = time.time()

        #Iterating through each sub dataframe inside main dataframe.
        for df_ in df:
        		#Log start time of parsing a sub dataframe of 2000 tweets.
                t3 = time.time()

                #Map all originator tweets to new dataframe.
                df_lst = dfa.loc[dfa["retweet_userid"].map(lambda x: str(x)==df_["userid"].apply(converter).unique().tostring())]

                #Write the new dataframe to a csv file
                df_lst.to_csv('my_parsed1.csv', mode='a', header=False)

                #Log end time of parsing the sub dataframe of 2000 tweets.
                t2 = time.time()
                #print total time completed to run iteration.
                print(t2-t3)

        #Log end time of parsing the sub dataframe of 2000 tweets.
        t1 = time.time()
        #print total time completed to run iteration.
        print(t1-t0)
