import pandas as pd
import numpy as np
from os import path

#Read File of all the unique userids of the bots that tweeted messages regarding to the 2016 elections.
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/ListIDS.csv", sep="\n", skiprows=[0], header=None, usecols=[0], names=["userid"])

loc=[]  #List that contains the above variables for each bot in a form of a class respectively.
tnt = 0	#Total no. of tweets read.

#Iteratring through each bot userid from above
for i in dfn.userid:
	fsn = "/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Bot-"+str(i)+"-EN.csv"
	if(path.isfile(fsn)==True):	#If such a bots csv file exists in the current directory
		#Setup the dataframe that reads each bot's tweet csv file.
		dfi = pd.read_csv(fsn, sep=",", header=None, names=["tweetid", "userid","tweet_candidate_class", "tweet_sentiment_class"])

		#Initialize the variables, that represent the no. of tweets belonging to a certain class and sentiment respectively, every time:
		nOT = 0 #No. of Trump class tweets for a certain bot.
		nOC = 0 #No. of Clinton class tweets for a certain bot.
		nON = 0 #No. of Neutral class tweets for a certain bot.
		nop = 0 #No. of positive sentiment tweets for a certain bot.
		non = 0 #No. of negative sentiment tweets for a certain bot.
		noN = 0 #No. of neutral sentiment tweets for a certain bot.
		npT = 0 #No. of positive Trump tweets for a certain bot.
		nnT = 0 #No. of negative Trump tweets for a certain bot.
		nNT = 0 #No. of neutral Trump tweets for a certain bot.
		npC = 0 #No. of positive Clinton tweets for a certain bot.
		nnC = 0 #No. of negative Clinton tweets for a certain bot.
		nNC = 0 #No. of neutral Clinton tweets for a certain bot.
		tnb = 0 #Total no. of tweets for each bot.

		for i in dfi.index:
			tnb += 1
			tnt += 1

			if(dfi["tweet_candidate_class"][i]=="Trump"):
				nOT += 1
				if(dfi["tweet_sentiment_class"][i]=="Positive"):
					nop += 1
					npT += 1
				if(dfi["tweet_sentiment_class"][i]=="Negative"):
					non += 1
					nnT += 1
				if(dfi["tweet_sentiment_class"][i]=="Neutral"):
					noN += 1
					nNT += 1
			if(dfi["tweet_candidate_class"][i]=="Clinton"):
				nOC += 1
				if(dfi["tweet_sentiment_class"][i]=="Positive"):
					nop += 1
					npC += 1
				if(dfi["tweet_sentiment_class"][i]=="Negative"):
					non += 1
					nnC += 1
				if(dfi["tweet_sentiment_class"][i]=="Neutral"):
					noN += 1
					nNC += 1
			if(dfi["tweet_candidate_class"][i]=="Neutral"):
				nON += 1
				noN += 1
		loc.append([(nOT/tnb)*100, (nOC/tnb)*100, (nON/tnb)*100, (nop/tnb)*100, (non/tnb)*100, (noN/tnb)*100, (npT/tnb)*100, (nnT/tnb)*100, (nNT/tnb)*100, (npC/tnb)*100, (nnC/tnb)*100, (nNC/tnb)*100, tnb])

apT = 0	#Overall average demographic of ProTrump tweets of each bot.
anT = 0 #Overall average demographic of AntiTrump tweets of each bot.
apC = 0 #Overall average demographic of ProClinton tweets of each bot.
anC = 0 #Overall average demographic of AntiClinton tweets of each bot.
aop = 0 #Overall average demographic of positivity tweets of each bot.
aon = 0 #Overall average demographic of negativity tweets of each bot.
aoT = 0 #Overall average demographic of tweets of each bot that are just about Trump.
aoC = 0 #Overall average demographic of tweets of each bot that are just about Clinton.
anN = 0 #Overall average demographic of tweets of each bot that are just Neutral.
aoN = 0 #Overall average demographic of neutrality tweets of each bot.

tpT = 0 #Total ProTrump tweets.
tnT = 0 #Total AntiTrump tweets.
tpC = 0 #Total ProClinton tweets.
tnC = 0 #Total AntiClinton tweets.
top = 0 #Total positivity tweets.
ton = 0 #Total negativity tweets.
toT = 0 #Total Trump tweets.
toC = 0 #Total Hillary tweets.
tnN = 0 #Total Neutral tweets.
toN = 0 #Total neutrality tweets.

for i in loc:
	print("The % of Trump tweets of this bot = " + str(i[0]) +"\n" +"The % of Clinton tweets of this bot = " + str(i[1]) +"\n" +"The % of Neutral tweets of this bot = " + str(i[2]) +"\n" +"The % of positive sentiment tweets of this bot = " + str(i[3]) +"\n" +"The % of negative sentiment tweets of this bot = " + str(i[4]) +"\n" +"The % of neutral sentiment tweets of this bot = " + str(i[5]) +"\n" +"The % of ProTrump tweets of this bot = " + str(i[6]) +"\n" +"The % of AntiTrump tweets of this bot = " + str(i[7]) +"\n" +"The % of Neutral Trump tweets of this bot = " + str(i[8]) +"\n" + "The % of ProClinton tweets of this bot = " + str(i[9]) +"\n" + "The % of AntiClinton tweets of this bot = " + str(i[10]) +"\n" + "The % of Neutral Clinton tweets of this bot = " + str(i[11]) +"\n")
	if(i[6]>0):
		apT += i[6]*(i[12])
		tpT += i[12]
	if(i[7]>0):
		anT += i[7]*(i[12])
		tnT += i[12]
	if(i[9]>0):
		apC += i[9]*(i[12])
		tpC += i[12]
	if(i[10]>0):
		anC += i[10]*(i[12])
		tnC += i[12]
	if(i[3]>0):
		aop += i[3]*(i[12])
		top += i[12]
	if(i[4]>0):
		aon += i[4]*(i[12])
		ton += i[12]
	if(i[0]>0):
		aoT += i[0]*(i[12])
		toT += i[12]
	if(i[1]>0):
		aoC += i[1]*(i[12])
		toC += i[12]
	if(i[2]>0):
		anN += i[2]*(i[12])
		tnN += i[12]
	if(i[5]>0):
		aoN += i[5]*(i[12])
		toN += i[12]

print("#############################Average Demographic of each bot:##########################")
try:
	print("If a bot represents ProTrump, on average, the % of its tweets that are ProTrump = "+str(apT/tpT)+"%")
except ZeroDivisionError:
	print("If a bot represents ProTrump, on average, the % of its tweets that are ProTrump = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents AntiTrump, on average, the % of its tweets that are AntiTrump = "+str(anT/tnT)+"%")
except ZeroDivisionError:
	print("If a bot represents AntiTrump, on average, the % of its tweets that are AntiTrump = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents ProHillary, on average, the % of its tweets that are ProHillary = "+str(apC/tpC)+"%")
except ZeroDivisionError:
	print("If a bot represents ProHillary, on average, the % of its tweets that are ProHillary = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents AntiHillary, on average, the % of its tweets that are AntiHillary = "+str(anC/tnC)+"%")
except ZeroDivisionError:
	print("If a bot represents AntiHillary, on average, the % of its tweets that are AntiHillary = DID NOT FIND ANY SUCH TWEETS")
try:
        print("If a bot represents ProTrump, on average, the % of its tweets that are AntiHillary = "+str(anC/tpT)+"%")
except ZeroDivisionError:
        print("If a bot represents ProTrump, on average, the % of its tweets that are AntiHillary = DID NOT FIND ANY SUCH TWEETS")
try:
        print("If a bot represents AntiTrump, on average, the % of its tweets that are ProHillary = "+str(apC/tnT)+"%")
except ZeroDivisionError:
        print("If a bot represents AntiTrump, on average, the % of its tweets that are ProHillary = DID NOT FIND ANY SUCH TWEETS")
try:
        print("If a bot represents ProHillary, on average, the % of its tweets that are AntiTrump = "+str(anT/tpC)+"%")
except ZeroDivisionError:
        print("If a bot represents ProHillary, on average, the % of its tweets that are AntiTrump = DID NOT FIND ANY SUCH TWEETS")
try:
        print("If a bot represents AntiHillary, on average, the % of its tweets that are ProTrump = "+str(apT/tnC)+"%")
except ZeroDivisionError:
        print("If a bot represents AntiHillary, on average, the % of its tweets that are ProTrump = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents positivity in general, on average, the % of its tweets that are positive = "+str(aop/top)+"%")
except ZeroDivisionError:
	print("If a bot represents positivity in general, on average, the % of its tweets that are positive = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents negativity in general, on average, the % of its tweets that are negative = "+str(aon/ton)+"%")
except ZeroDivisionError:
	print("If a bot represents negativity in general, on average, the % of its tweets that are negative = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents solely Trump speech, on average, the % of its tweets that are about Trump = "+str(aoT/toT)+"%")
except ZeroDivisionError:
	print("If a bot represents solely Trump speech, on average, the % of its tweets that are about Trump = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents solely Hillary speech, on average, the % of its tweets that are about Hillary = "+str(aoC/toC)+"%")
except ZeroDivisionError:
	print("If a bot represents solely Hillary speech, on average, the % of its tweets that are about Hillary = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents a Neutral stance in the elections, on average, the % of its tweets that are neutral = "+str(anN/tnN)+"%")
except ZeroDivisionError:
	print("If a bot represents a Neutral stance in the elections, on average, the % of its tweets that are neutral = DID NOT FIND ANY SUCH TWEETS")
try:
	print("If a bot represents neutrality, on average, the % of its tweets that are neutrality = "+str(aoN/toN)+"%")
except ZeroDivisionError:
	print("If a bot represents neutrality, on average, the % of its tweets that are neutrality = DID NOT FIND ANY SUCH TWEETS")
