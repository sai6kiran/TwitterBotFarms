import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import goslate
import numpy as np
import cld3
from googletrans import Translator

###############DataFrames:###########################
dfn = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,2,3,6,12,14,15,16,19,20,27,28], chunksize=2000, names=["tweetid", "userid", "user_display_name", "user_screen_name", "user_profile_url", "tweet_text", "tweet_client_name", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid", "hashtags", "urls"])
dfi = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/kCoreBotsList.csv", sep=",", header=None, chunksize=2000)
df_lst = pd.DataFrame(columns=["tweetid", "userid", "user_display_name", "user_screen_name", "user_profile_url", "tweet_text", "tweet_client_name", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid", "hashtags", "urls"])
kcu = []

###########Defining our Google Translator:###########
#translator = Translator(to_lang='en', from_lang='ru')

##########Creating each Core Bot's csv file:#########
for df_ in dfi:
	for i in df_.values:
		kcu.append(i[0])

###########List of Words Used to detect if tweet is a Political Tweet:############################
eng=["Hillary", "Trump", "debate", "president", "U.S.", "america", "russia", "election", "poll", "Clinton", "Donald", "Putin", "Vladmir", "parliament", "constitution", "democracy", "rump"]
rus=["Hillary", "козырной", "обсуждение", "президент", "НАС", "Америка", "Россия","выборы", "опрос", "Клинтон", "Дональд", "Путин", "Владимир", "Лари", "парламент", "конституции", "демократия", "крестец"]

################Streaming All English Political Tweets of each bots into the bots own .CSV file:###################
for df_ in dfn:
	df_lst.iloc[0:0]
	df_lst = df_.loc[df_["userid"].map(lambda x: x in kcu)]
	df_lst = df_lst.loc[df_lst["tweet_text"].apply(lambda x: cld3.get_language(str(x)).language=='en')]	#Obtain All English Tweets

	for z in eng:
		df_lst["Yes"] = df_["tweet_text"].apply(lambda x: "true" if str(z).lower() in str(x).lower() and cld3.get_language(str(x)).language=='en' else "false")	#Obtain all English tweets that are Political.
	df_lst = df_lst.loc[df_lst["Yes"].map(lambda x: x=="true"]

	df_lst.insert(14, "language", "en")
	df_lst = df_lst.dropna(subset=["tweet_text"])
	df_lst[["tweetid", "userid", "tweet_text", "hashtags", "urls", "language"]].to_csv("EnglishPoliticalTweets.csv", mode='a', header=False, index=False)
	df_lst[["tweetid", "userid", "tweet_text"]].to_csv(fsn, mode='a', header=False, index=False)
