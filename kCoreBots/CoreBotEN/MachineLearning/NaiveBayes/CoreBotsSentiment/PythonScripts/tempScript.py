import pandas as pd
import numpy as np
import csv

dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/ListIDS.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])

#Dictionary that stores lists used to calculate demographic statistics below:
pbd = {}        #Political Bot Dictionary. I.e. Dictionary of all twitter bots that tweeted, replied to, or retweeted political comments that affected the 2016 elections. The key represents the bot's userid. The value is a l$
toc = ""
nmc = 0 #Total No. of bots that represent multiple classes. I.e. Have multiple sentiments or are targetting multiple candidates.
npn = 0 #Total No. of bots that are both positive and negative in sentimentality.
ntc = 0 #Total No. of bots that target both Trump and Clinton.
nPtAc = 0       #Total No. of bots that are Pro Trump and Anti Clinton.
nPtAt = 0       #Total No. of bots that are Pro Trump and Anti Trump.
nAtPc = 0       #Total No. of bots that are Anti Trump and Pro Clinton.
nPcAc = 0       #Total No. of bots that are Pro Clinton and Anti Clinton.
nPtPc = 0       #Total No. of bots that are Pro Trump and Pro Clinton.
nAtAc = 0       #Total No. of bots that are Anti Trump and Anti Clinton.

#Dataset required to classify each tweet and its sentimentality to its corresponding bot:
dfc = pd.DataFrame(columns=["tweetid", "userid", "tweet_candidate_class", "tweet_sentiment_class"])

for i in dfn.index:
	fsn = "/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Bot-"+str(dfn["userid"][i])+"-EN.csv"

	#Setup the dataframe that reads each bot's tweet csv file.
	dfi = pd.read_csv(fsn, sep=",", header=None, names=["tweetid","userid","tweet_candidate_class","tweet_sentiment_class"])
	for j in dfi.index:
		if(str(dfi["tweet_candidate_class"][j]).lower()=="trump" and str(dfi["tweet_sentiment_class"][j]).lower()=="positive"):
			toc = "ProTrump"
		elif(str(dfi["tweet_candidate_class"][j]).lower()=="trump" and str(dfi["tweet_sentiment_class"][j]).lower()=="negative"):
			toc = "AntiTrump"
		elif(str(dfi["tweet_candidate_class"][j]).lower()=="clinton" and str(dfi["tweet_sentiment_class"][j]).lower()=="positive"):
			toc = "ProClinton"
		elif(str(dfi["tweet_candidate_class"][j]).lower()=="clinton" and str(dfi["tweet_sentiment_class"][j]).lower()=="negative"):
			toc = "AntiClinton"
		else:
			toc = "Neutral"

		tmp = [dfi["tweet_candidate_class"][j], dfi["tweet_sentiment_class"][j], toc]   #Temporary List

		if(dfi.iloc[j].userid in pbd.keys()):
			if(tmp not in pbd[dfi.iloc[j].userid]):
				tvl = dfi.iloc[j].userid        #temporary value
				pbd[tvl]=pbd[tvl]+[tmp]
		else:
			pbd[dfi.iloc[j].userid] = [tmp]

for key, val in pbd.items():
	if(len(val)>1):
		nmc += 1
	if(any("Positive" in all for all in val) and any("Negative" in all for all in val)):
		npn += 1
	if(any("Trump" in all for all in val) and any("Clinton" in all for all in val)):
		ntc += 1
	if(any("ProTrump" in all for all in val) and any("AntiClinton" in all for all in val)):
		nPtAc += 1
	if(any("ProTrump" in all for all in val) and any("AntiTrump" in all for all in val)):
		nPtAt += 1
	if(any("AntiTrump" in all for all in val) and any("ProClinton" in all for all in val)):
		nAtPc += 1
	if(any("ProClinton" in all for all in val) and any("AntiClinton" in all for all in val)):
		nPcAc += 1
	if(any("ProTrump" in all for all in val) and any("ProClinton" in all for all in val)):
		nPtPc += 1
	if(any("AntiTrump" in all for all in val) and any("AntiClinton" in all for all in val)):
		nAtAc += 1

#Oprint(pbd)
print("*****************General demographics of the bots:*********************")
print("Total no. of bots that have multiple classes = " +str(nmc))
print("Total no. of bots that are both positive and neagtive in sentimentality = " +str(npn))
print("Total no. of bots that target both Trump and Hillary = " +str(ntc))
print("Total no. of bots that are both ProTrump and AntiClinton = " +str(nPtAc))
print("Total no. of bots that are both ProTrump and AntiTrump = " +str(nPtAt))
print("Total no. of bots that are both AntiTrump and ProClinton = " +str(nAtPc))
print("Total no. of bots that are both ProClinton and AntiClinton = " +str(nPcAc))
print("Total no. of bots that are both ProTrump and ProClinton = " +str(nPtPc))
print("Total no. of bots that are both AntiTrump and AntiClinton = " +str(nAtAc))
