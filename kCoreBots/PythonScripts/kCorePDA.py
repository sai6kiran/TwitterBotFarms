import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import goslate
import numpy as np
import cld3
from googletrans import Translator

###############DataFrames:#########################
dfn = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,2,3,6,12,14,15,16,19,20,27,28], chunksize=2000, names=["tweetid", "userid", "user_display_name", "user_screen_name", "user_profile_url", "tweet_text", "tweet_client_name", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid", "hashtags", "urls"])
dfi = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/kCoreBotsList.csv", sep=",", header=None, chunksize=2000)
df_lst = pd.DataFrame(columns=["tweetid", "userid", "user_display_name", "user_screen_name", "user_profile_url", "tweet_text", "tweet_client_name", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid", "hashtags", "urls"])

############Defining our Google Translator:##################
#translator = Translator(to_lang='en', from_lang='ru')

############Creating each Core Bot's csv file:###############
kcu = []

for df_ in dfi:
	for i in df_.values:
		kcu.append(i[0])

###########Stream the list of Political Tweets and Tweets of Highest Order Core bots of each language into either English (EN) or Russian (RU) .CSV file:######################3
cnt = 0
for i in kcu:
	cnt+=1
	fsn = "Bot"+str(cnt)+"EN.csv"
	dff = pd.DataFrame([["The", " Bot UserID", " is:", " "+str(i)]], columns=["tweetid", "tweet_text", "hashtags", "urls"])

	for df_ in dfn:
		#translators = Translator(to_lang='en', from_lang='ru')	#Convert Russian tweets into English.
		df_lst.iloc[0:0]
		df_lst = df_.loc[df_["userid"].map(lambda x: x==i)]
		for z in eng:
			df_lst["Yes"] = df_lst["tweet_text"].apply(lambda x: "true" if z in x else "false")
		df_lst = df_lst.loc[df_lst["Yes"].map(lambda x: x=="true")]
		df_lst[["tweetid", "tweet_text", "hashtags", "urls"]].to_csv(fsn, mode='a', header=False, index=False)

