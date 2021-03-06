import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import goslate
import numpy as np
import cld3
#from googletrans import Translator

dfn = pd.read_csv("/root/.encrypted/.pythonSai/ira_tweets_csv_hashed.csv", sep=",", header=None, usecols=[0,1,2,3,6,12,14,15,16,19,20,27,28], chunksize=2000, names=["tweetid", "userid", "user_display_name", "user_screen_name", "user_profile_url", "tweet_text", "tweet_client_name", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid", "hashtags", "urls"])
dfi = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/kCoreBotsList.csv", sep=",", header=None, chunksize=2000)
df_lst = pd.DataFrame(columns=["tweetid", "userid", "user_display_name", "user_screen_name", "user_profile_url", "tweet_text", "tweet_client_name", "in_reply_to_tweetid", "in_reply_to_userid", "retweet_userid", "retweet_tweetid", "hashtags", "urls"])
kcu = []

#Defining our Google Translator:
#translator = Translator(to_lang='en', from_lang='ru')

#Creating each Core Bot's csv file:
for df_ in dfi:
	#t0 = time.time()
	for i in df_.values:
		kcu.append(i[0])


#dfb = pd.DataFrame(data={"col1": kcu})
#dfb.to_csv("test.csv", sep=',',index=False)
import pdb
cnt = 0
eng=["Hillary", "Trump", "debate", "president", "U.S.", "america", "russia", "election", "poll", "Clinton", "Donald", "Putin", "Vladmir", "lary", "parliament", "constitution", "democracy", "rump"]
rus=["Hillary", "Trump", "U.S.", "america", "Clinton", "Donald", "Vladimir", "Putin", "козырной", "обсуждение", "президент", "НАС", "Америка", "Россия","выборы", "опрос", "Клинтон", "Дональд", "Путин", "Владимир", "Лари", "парламент", "конституции", "демократия", "крестец"]

#for i in kcu:
	#cnt+=1
fsn = "CoreBotTweetsCombinedRU.csv"
	#dff = pd.DataFrame([["The", " Core Bot UserID", " is:", " "+str(i)]], columns=["tweetid", "tweet_text", "hashtags", "urls"])
	#dff[["tweet_text"]] = dff["tweet_text"].apply(translator.translate, dest='en').apply(getattr, args=('text',))
	#dff[["tweetid", "tweet_text", "hashtags", "urls"]].to_csv(fsn, mode='a', header=i, index=False)
for df_ in dfn:
		#translators = Translator(to_lang='en', from_lang='ru')	
	df_lst.iloc[0:0]
		#gs = goslate.Goslate()
	t0 = time.time()
		#print(df_.loc[df_["userid"].map(lambda x: x==i)]["tweet_text"])
	df_lst = df_.loc[df_["userid"].map(lambda x: x is not None)]
	for z in rus:
		df_lst["Yes"] = df_["tweet_text"].apply(lambda x: "true" if str(z).lower() in str(x).lower() and cld3.get_language(str(x)).language=='ru' else "false")
	df_lst = df_lst.loc[df_lst["Yes"].map(lambda x: x=="true")]
                #pdb.set_trace()
                #print(df_.tweet_text)
                #print(df_["tweet_text"].apply(lambda x: translators.translate(x)))
                #df_lst["tweet_text"] = df_lst.tweet_text.apply(lambda x: gs.translate(x, 'en')
	df_lst.insert(14, "language", "ru")
	df_lst = df_lst.dropna(subset=["tweet_text"])

	df_lst[["tweetid", "tweet_text", "hashtags", "urls", "language"]].to_csv(fsn, mode='a', header=False, index=False)
	t1 = time.time()
	print(t1-t0)
