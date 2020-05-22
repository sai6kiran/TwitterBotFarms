from contractionsDict import contractionsDict
import pandas as pd
import time
import numpy as np
import re
from pattern.en import pluralize, singularize
import sys
import csv
from LemmitizationandStemConverter import ObtainStemAndLemmatizationWord

def priorProb(scv):
	pct = 0	#positive count total
	nct = 0	#negative count total
	Nct = 0	#neutral count total
	ntt = 0	#no. training tweets
	for index, row in scv.items():
		#print(row)
		if(row.lower() == 'positive'):
			pct+=1
		if(row.lower() == 'negative'):
			nct+=1
		if(row.lower() == 'neutral'):
			Nct+=1
		ntt+=1
	pc1 = pct/ntt	#Postive Class 1
	nc2 = nct/ntt	#Negative Class 2
	nc3 = Nct/ntt	#Neutral Class 3
	return((pc1, nc2, nc3))

def removeEmojis(txt):
	emoji_pattern = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)
	return(emoji_pattern.sub(u' ', txt))

def expandContractions(s, contractionsDict=contractionsDict):
	contractionsRe = re.compile('(%s)' % '|'.join(contractionsDict.keys()))
	def replace(match):
		return contractionsDict[match.group(0)]
	return contractionsRe.sub(replace, s)

def CleanUp(text):
	#Removes links from tweet:
	text = re.sub('http://\S+|https://\S+', ' ', text)

	#Remove #, _, -, and @ from tweet:
	text = text.replace("#", " ").replace("_", " ").replace("@", " ").replace("-", " ")

	#Replace ? with questionmark and ! with exclaimationmark:
	text = text.replace("?", " questionmark").replace("!", " exclaimationmark")

	#Remove all other non alphanumeric special characters from tweet:
	text = re.sub('\W+ ',' ', text)

	#Removes whitespaces from tweet:
	text = text.replace("\t", " ").replace("\n", " ")
	text = re.sub(r' {2,}' , ' ', text)

	#Removes emojis from tweet:
	text = removeEmojis(text)

	return text


def likelihoodFunctionInformation(txt, ldf):
	tsv  = 0	#Total Sentiment Value
	npw = 0		#No. of positive words
	nnw = 0		#No. negative words
	nNw = 0		#No. of neutral words

	psv = 0		#Previous Word sentiment value
	nac = False	#Negative conjuctive Adverb check
	wrd = " "	#Word to parse
	t3 = time.time()
	for ewt in txt.split():

		'''
		#Check for all versions of word in Sentiment Dictionary:
		#print(ewt)
		#t1 = time.time()
		sll = ObtainStemAndLemmatizationWord(ewt)       #Obtaining the noun version and root version of word using the function.
		#print(sll)
		if(sll[0]!=ewt):
			if(bool(sll[0] and sll[0].strip())==True):	#Checing if the noun part of the word is in the Sentiment Dictionary.
				snw = singularize(sll[0])  #Noun part of word in singular tense.
				pnw = pluralize(sll[0])    #Noun part of word in plural tense.
				srw = singularize(sll[1])  #Root part of word in singular tense.
				prw = pluralize(sll[1])    #Root part of word in plural tense.
				#Check if singular part of noun of word is in the Sentiment Dictionary:
				if((snw in ldf[0].word.values) or (snw in ldf[1].word.values) or (snw in ldf[2].word.values) or (snw in ldf[3].word.values)):
					wrd = snw
				#Check if plural part of noun of word is in the Sentiment Dictionary:
				elif((pnw in ldf[0].word.values) or (pnw in ldf[1].word.values) or (pnw in ldf[2].word.values) or (pnw in ldf[3].word.values)):
					wrd = pnw
				#Check if singular part of root of word is in the Sentiment Dictionary:
				elif((srw in ldf[0].word.values) or (srw in ldf[1].word.values) or (srw in ldf[2].word.values) or (srw in ldf[3].word.values)):
					wrd = srw
				#Check if plural part of root of word is in the Sentiment Dictionary:
				elif((prw in ldf[0].word.values) or (prw in ldf[1].word.values) or (prw in ldf[2].word.values) or (prw in ldf[3].word.values)):
					wrd = prw
				else:
					wrd = ewt
			elif(sll[1]!=ewt):	#Checking if the root version of the word is in the Sentiment Dictionary.
				srw = singularize(sll[1])  #Root part of word in singular tense.
				prw = pluralize(sll[1])    #Root part of word in plural tense.
				#Check if singular part of root of word is in the Sentiment Dictionary:
				if((srw in ldf[0].word.values) or (srw in ldf[1].word.values) or (srw in ldf[2].word.values) or (srw in ldf[3].word.values)):
					wrd = srw
				#Check if plural part of root of word is in the Sentiment Dictionary:
				elif((prw in ldf[0].word.values) or (prw in ldf[1].word.values) or (prw in ldf[2].word.values) or (prw in ldf[3].word.values)):
					wrd = prw
				else:
					wrd = ewt
			else:
				wrd = ewt
		else:
			wrd = ewt
		'''
		wrd = ewt

		#Run the Likelihood Function Information on the word.
		wsv = 0	#Word Sentiment Value
		sfw = singularize(wrd)	#Singular Form of Word
		pfw = pluralize(wrd)	#Plural Form of Word
		#print(wrd, tsv)	#Very Important Print Statement for Debugging

		#Checking if word matches a negative conjuctive adverb that forms different phrases in the tweet:
		if wrd.lower()=='not' or wrd.lower()=='but' or wrd.lower()=='however' or wrd.lower()=='instead' or wrd.lower()=='otherwise' or wrd.lower()=='contrarily':
			if(nac==False):
				nac=True
			else:
				nac=False
		if(nac==False):
			#Checking if words match special words
			if sfw.lower()=='maga':
				npw += 100
				tsv += 100
			elif sfw.lower()=='makeamericagreatagain':
				npw += 100
				tsv += 100
			elif sfw.lower()=='make america great again':
				npw += 100
				tsv += 100
			elif "email" in sfw.lower():
				nnw += 5
				tsv -= 5
			elif wrd.lower()=='questionmark':
				if(psv>0):
					nnw += 10
					tsv -= 10
				if(psv<0):
					npw += 10
					tsv += 10
				psv = 0
			elif wrd.lower()=='exclaimationmark':
				if(psv<0):
					nnw += 10
					tsv -= 10
				if(psv>0):
					npw += 10
					tsv += 10
				psv = 0

			#Checking if word exists in the Sentiment Dictionary. Assign sentiment value and/or category if word exists. Otherwise categorize word as neutral.
			elif sfw.lower() in ldf[0].word.values:	#Check if singular version of word is in dataframe1
				wsv = int(ldf[0].iloc[ldf[0]['word'].loc[lambda x: x==sfw.lower()].index.tolist()[0]].sentiment)
				#print(ewt, sfw, 1, wsv, tsv)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif pfw.lower() in ldf[0].word.values:	#Check if plural version of word is in dataframe1
				wsv = int(ldf[0].iloc[ldf[0]['word'].loc[lambda x: x==pfw.lower()].index.tolist()[0]].sentiment)
				#print(ewt, pfw, 1, wsv, tsv)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif sfw.lower() in ldf[1].word.values:	#Check if singular version of word is in dataframe2
				#print(ewt, sfw, 2)
				wsv = int(ldf[1].iloc[ldf[1]['word'].loc[lambda x: x==sfw.lower()].index.tolist()[0]].sentiment)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif pfw.lower() in ldf[1].word.values:	#Check if plural version of word is in dataframe2
				#print(ewt, pfw, 2)
				wsv = int(ldf[1].iloc[ldf[1]['word'].loc[lambda x: x==pfw.lower()].index.tolist()[0]].sentiment)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif sfw.lower() in ldf[2].word.values:	#Check if singular version of word is in dataframe3
				#print(ewt, sfw, 3, tsv)
				npw += 1
				psv = 3
			elif pfw.lower() in ldf[2].word.values:	#Check if plural version of word is in dataframe3
				#print(ewt, pfw, 3, tsv)
				npw += 1
				psv = 3
			elif sfw.lower() in ldf[3].word.values:	#Check if singular version of word is in dataframe4
				#print(ewt, sfw, 4)
				nnw += 1
				psv = -3
			elif pfw.lower() in ldf[3].word.values:	#Check if plural version of word is in dataframe4
				#print(ewt, pfw, 4)
				nnw += 1
				psv = -3
			else:					#The word must be a "neutral" word
				#print(wrd, sfw, pfw)
				nNw += 1
		else:
			#Checking if words match special words
			if sfw.lower()=='maga':
				npw += 100
				tsv += 100
			elif sfw.lower()=='makeamericagreatagain':
				npw += 100
				tsv += 100
			elif sfw.lower()=='make america great again':
				npw += 100
				tsv += 100
			elif "email" in sfw.lower():
				nnw += 5
				tsv -= 5
			elif wrd.lower()=='questionmark':
				if(psv>0):
					npw += 10
					tsv += 10
				if(psv<0):
					nnw += 10
					tsv -= 10
				psv = 0
				nac==False
			elif wrd.lower()=='exclaimationmark':
				if(psv<0):
					npw += 10
					tsv += 10
				if(psv>0):
					nnw += 10
					tsv -= 10
				psv = 0
				nac==False

                        #Checking if word exists in the Sentiment Dictionary. Assign sentiment value and/or category if word exists. Otherwise categorize word as neutral.
			elif sfw.lower() in ldf[0].word.values: #Check if singular version of word is in dataframe1
				wsv = int(ldf[0].iloc[ldf[0]['word'].loc[lambda x: x==sfw.lower()].index.tolist()[0]].sentiment)
				#print(sfw, 1, wsv, tsv)
				if(wsv>0):
					nnw += 1
				elif(wsv<0):
					npw += 1
				tsv -= wsv
				psv = -wsv
				nac=False
			elif pfw.lower() in ldf[0].word.values: #Check if plural version of word is in dataframe1
				wsv = int(ldf[0].iloc[ldf[0]['word'].loc[lambda x: x==pfw.lower()].index.tolist()[0]].sentiment)
				#print(pfw, 1, wsv, tsv)
				if(wsv>0):
					nnw += 1
				elif(wsv<0):
					npw += 1
				tsv -= wsv
				psv = -wsv
				nac==False
			elif pfw.lower() in ldf[0].word.values: #Check if plural version of word is in dataframe1
				wsv = int(ldf[0].iloc[ldf[0]['word'].loc[lambda x: x==pfw.lower()].index.tolist()[0]].sentiment)
				#print(pfw, 1, wsv, tsv)
				if(wsv>0):
					npw -= 1
				elif(wsv<0):
					nnw -= 1
				tsv -= wsv
				psv = -wsv
				nac==False
			elif sfw.lower() in ldf[1].word.values: #Check if singular version of word is in dataframe2
				#print(sfw, 2)
				wsv = int(ldf[1].iloc[ldf[1]['word'].loc[lambda x: x==sfw.lower()].index.tolist()[0]].sentiment)
				if(wsv>0):
					nnw += 1
				elif(wsv<0):
					npw += 1
				tsv -= wsv
				psv = -wsv
				nac==False
			elif pfw.lower() in ldf[1].word.values: #Check if plural version of word is in dataframe2
				#print(pfw, 2)
				wsv = int(ldf[1].iloc[ldf[1]['word'].loc[lambda x: x==pfw.lower()].index.tolist()[0]].sentiment)
				if(wsv>0):
					nnw += 1
				elif(wsv<0):
					npw += 1
				tsv -= wsv
				psv = -wsv
				nac==False
			elif sfw.lower() in ldf[2].word.values: #Check if singular version of word is in dataframe3
				#print(sfw, 3, tsv)
				nnw += 1
				psv = -3
				nac==False
			elif pfw.lower() in ldf[2].word.values: #Check if plural version of word is in dataframe3
				#print(pfw, 3, tsv)
				nnw += 1
				psv = -3
				nac==False
			elif sfw.lower() in ldf[3].word.values: #Check if singular version of word is in dataframe4
				#print(sfw, 4)
				npw += 1
				psv = 3
				nac==False
			elif pfw.lower() in ldf[3].word.values: #Check if plural version of word is in dataframe4
				#print(pfw, 4)
				npw += 1
				psv = 3
				nac==False
			else:                                   #The word must be a "neutral" word
				#print(wrd, sfw, pfw)
				nNw += 1
		#t2 = time.time()
		#print("Amount of time taken to parse word: " + str(t2-t1) + "sec")

	t4 = time.time()
	print("Amount of time taken to parse tweet: " + str(t4-t3) + "sec")
	return(npw, nnw, nNw, tsv)

def NaiveBayes(txt, ppl, tov):
	#tov = likelihoodFunctionInformation(ctt, [df1, df2, df3, df4])	#Obtain tuple of values required to calculate the Likelihood funnction and posterior probability
	pPp = ppl[0]	#Positive class Prior Probability
	pnp = ppl[1]	#Negative class Prior Probability
	pNp = ppl[2]	#Neutral class Prior Probability
	npw = tov[0]	#No. of positive words
	nnw = tov[1]	#No. of negative words
	nNw = tov[2]	#No. of neutral words
	tsv = tov[3]	#Total Sentiment Value
	tnw = npw + nnw + nNw	#Total no. of words
	cls = " "	#Defining the class which the text belongs to.

	#print(npw,  nnw, nNw, tsv)
	if(npw==0 and nnw==0):
		cls = "neutral"	#Class is set to Neutral
	else:
		if(tsv==0):
			den = (pPp*(1-np.exp(-1*((npw*5)/(tnw))))) + (pnp*(1-np.exp(-1*((nnw*5)/(tnw))))) + (pNp*(1-np.exp(-1*((nNw)/(tnw)))))	#Calculate the denominator for the posterior probabilities

			#Posterior Probability of sentiment of text is positive given the text:
			ppp = (pPp*(1-np.exp(-1*((npw*5)/(tnw)))))/(den)
			#print((1-np.exp(-1*(npw*10))))
			#print(ppp)

			#Posterior Probability of sentiment of text is negative given the text:
			npp = (pnp*(1-np.exp(-1*((nnw*5)/(tnw)))))/(den)
			#print((1-np.exp(-1*(nnw*10))))
			#print(npp)

			#Posterior Probability of sentiment of text is neutral given the text:
			Npp = (pNp*(1-np.exp(-1*((nNw)/(tnw)))))/(den)
			#print((1-np.exp(-1*(nNw*10))))
			#print(Npp)

			#Determine the sentimentality of text:
			if(max([ppp,npp,Npp])==ppp):
				cls = "positive"
			if(max([ppp,npp,Npp])==npp):
				cls = "negative"
			if(max([ppp,npp,Npp])==Npp):
				cls = "neutral"
		elif(tsv>0):
			den = (pPp*(1-np.exp(-1*((npw*5*tsv)/(tnw))))) + (pnp*(1-np.exp(-1*((nnw*5)/(tnw))))) + (pNp*(1-np.exp(-1*((nNw)/(tnw*1.45)))))        #Calculate the denominator for the posterior probabilities.

			#Posterior Probability of sentiment of text is positive given the text:
			ppp = (pPp*(1-np.exp(-1*((npw*5*tsv)/(tnw)))))/(den)
			#print((1-np.exp(-1*(npw*10))))
			#print(ppp)

			#Posterior Probability of sentiment of text is negative given the text:
			npp = (pnp*(1-np.exp(-1*((nnw*5)/(tnw)))))/(den)
			#print((1-np.exp(-1*(nnw*10))))
			#print(npp)

			#Posterior Probability of sentiment of text is neutral given the text:
			Npp = (pNp*(1-np.exp(-1*((nNw)/(tnw*1.45)))))/(den)
			#print((1-np.exp(-1*(nNw*10))))
			#print(Npp)

			#Determine the sentimentality of text:
			if(max([ppp,npp,Npp])==ppp):
				cls = "positive"
			if(max([ppp,npp,Npp])==npp):
				cls = "negative"
			if(max([ppp,npp,Npp])==Npp):
				cls = "neutral"
		else:
			den = (pPp*(1-np.exp(-1*((npw*5)/(tnw))))) + (pnp*(1-np.exp(-1*((nnw*5*abs(tsv))/(tnw))))) + (pNp*(1-np.exp(-1*((nNw)/(tnw*1.45)))))        #Calculate the denominator for the posterior probabilities.

			#Posterior Probability of sentiment of text is positive given the text:
			ppp = (pPp*(1-np.exp(-1*((npw*5*tsv)/(tnw)))))/(den)
			#print((1-np.exp(-1*(npw*10))))
			#print(ppp)

			#Posterior Probability of sentiment of text is negative given the text:
			npp = (pnp*(1-np.exp(-1*((nnw*5*abs(tsv))/(tnw)))))/(den)
			#print((1-np.exp(-1*(nnw*10))))
			#print(npp)

			#Posterior Probability of sentiment of text is neutral given the text:
			Npp = (pNp*(1-np.exp(-1*((nNw)/(tnw*1.45)))))/(den)
			#print((1-np.exp(-1*(nNw*10))))
			#print(Npp)

			#Determine the sentimentality of text:
			if(max([ppp,npp,Npp])==ppp):
				cls = "positive"
			if(max([ppp,npp,Npp])==npp):
				cls = "negative"
			if(max([ppp,npp,Npp])==Npp):
				cls = "neutral"
	return cls

#############Loading the Datasets:####################
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Training Dataset:
dft = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/trainingdataset.csv", sep=",", skiprows=[0], header=None, usecols=[0,1], names=["tweet_text","sentiment"])

#Testing Dataset:
dfT = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/testingdataset.csv", sep=",", skiprows=[0], header=None, usecols=[0,1], names=["tweet_text","sentiment"])

#Sample Dataset:
dfs =  pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/sampleDataset.csv", sep=",", skiprows=[0], header=None, usecols=[0,1,2], names=["tweetid", "userid", "tweet_text"])

#Main Dataset:
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/CoreBotTweetsCombinedEN.csv", sep=",", skiprows=[0], header=None, usecols=[0,1,2], names=["tweetid","userid", "tweet_text"])

#Sentiment Dataset 1:
df1 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/AFINN-111.txt", sep="\t", header=None, usecols=[0,1], names=["word","sentiment"])

#Sentiment Dataset 2:
df2 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/AFINN-96.txt", sep="\t", header=None, usecols=[0,1], names=["word","sentiment"])

#Sentiment Dataset 3 [Positive Words Only]:
df3 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/Positivewords.txt", sep="\n", header=None, usecols=[0], names=["word"])

#Sentiment Dataset 4 [Negative Words Only]:
df4 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/Negativewords.txt", sep="\n", header=None, usecols=[0], names=["word"])

#Dataset required to classify each tweet and its sentimentality to its corresponding bot:
dfc = pd.DataFrame(columns=["tweetid", "userid", "tweet_candidate_class", "tweet_sentiment_class"])


#############Running the Naive Bayesian Classifer:####################

#Obtain the list of Prior Probabilities obtained from Training Dataset:
tts = dft["sentiment"].count()	#Total no. of Training Sentiment values.
tTs = dfT["sentiment"].count()	#Total no. of Testing sentiment values.
#Append all the Testing sentiment values with the Training sentiment values to obtain a complete list of sentiments used as priorProbabalities for classification of all political tweets sent by "CoreBotTweetsCombinedEN.csv".
for i in range(tts, tts+tTs):
	dft["sentiment"][i] = dfT["sentiment"][i-tts]
ppl = priorProb(dft.sentiment)

loc = []	#List of classes for each text row in the dataframe.
#Dictionary that stores lists used to calculate demographic statistics below:
pbd = {}        #Political Bot Dictionary. I.e. Dictionary of all twitter bots that tweeted, replied to, or retweeted political comments that affected the 2016 elections. The key represents the bot's userid. The value is a list of class types it belongs to. i.e. Value = ["Trump", "positive", "ProTrump"].

for index, row in dfn.iterrows():
	#print(CleanUp(expandContractions(row["tweet_text"].replace("’", "'"))))
	ctt = CleanUp(expandContractions(row["tweet_text"].replace("’", "'")))	#Cleaned Tweet
	cot = NaiveBayes(ctt, ppl, likelihoodFunctionInformation(ctt, [df1, df2, df3, df4]))
	#print(cot)
	loc.append(cot)

tnr = 0	#Total No. of right words.
mcp = 0	#MisClassification percentage.
tap = 0	#Total Accuracy percentage.

npt = 0	#No. of positive Trump tweets.
nnt = 0	#No. of negative Trump tweets.
nNt = 0	#No. of neutral Trump tweets.
npc = 0	#No. of positive Clinton tweets.
nnc = 0	#No. of negative Clinton tweets.
nNc = 0	#No. of neutral Clinton tweets.
ngt = 0	#No. of general tweets. [i.e. Not Trump or Hillary].
tht = False	#Is the tweet a Trump or Hillary tweet?
tcc = " "	#Setting the tweet candidate class [i.e. Trump, Hillary, Neutral] for the classification below.
tsc = " "	#Setting the tweet sentiment class [i.e. Positive, Negative, Neutral] for the classification below.
toc = " "	#Setting the tweet overall class. [i.e. ProTrump, AntiClinton, etc;] for the classification below.

#t="RT @Trumpocrats: @TallahForTrump @tariqnasheed I'm beside myself by his hate for America and how we have done so much to free an entire rac..."
#print(t)
#print("Actual Sentiment: " + "negative")
#print("Calculated Sentiment: " + str(cot))


for i in range(0,len(loc)):
	#Recording no. of correct tweets:
	#print(dfn.iloc[i].tweet_text)
	#print("Actual Sentiment: " + dft.iloc[i].sentiment)
	#print("Calculated Sentiment: " + loc[i])
	'''
	if(loc[i].lower()==dft.iloc[i].sentiment.lower()):
		tnr += 1	#Use to calculate accuracy of classifier; Not for running entire algorithm
	'''
	#Classification of Tweets to Trump, Hillary or Neutral:
	if("trump" in dfn.iloc[i].tweet_text.lower() or "donald" in dfn.iloc[i].tweet_text.lower()):
		tht = True
		if(("email" in dfn.iloc[i].tweet_text.lower()) or ("makeamericagreatagain" in dfn.iloc[i].tweet_text.lower()) or ("make america great again" in dfn.iloc[i].tweet_text.lower()) or ("maga" in dfn.iloc[i].tweet_text.lower()) or ("russia" in dfn.iloc[i].tweet_text.lower())):
			npt += 1
			tcc = "Trump"
			tsc = "Positive"
			toc = "ProTrump"
		else:
			if(loc[i]=="positive"):
				npt += 1
				tcc = "Trump"
				tsc = "Positive"
				toc = "ProTrump"
			if(loc[i]=="negative"):
				nnt += 1
				tcc = "Trump"
				tsc = "Negative"
				toc = "AntiTrump"
			if(loc[i]=="neutral"):
				nNt += 1
				tcc = "Trump"
				tsc = "Neutral"
				toc = "Neutral"

	if("clinton" in dfn.iloc[i].tweet_text.lower() or "hillary" in dfn.iloc[i].tweet_text.lower()):
		tht = True
		if(("email" in dfn.iloc[i].tweet_text.lower()) or ("makeamericagreatagain" in dfn.iloc[i].tweet_text.lower()) or ("make america great again" in dfn.iloc[i].tweet_text.lower()) or ("maga" in dfn.iloc[i].tweet_text.lower()) or ("russia" in dfn.iloc[i].tweet_text.lower())):
			nnc += 1
			tcc = "Clinton"
			tsc = "Negative"
			toc = "AntiClinton"
		else:
			if(loc[i]=="positive"):
				npc += 1
				tcc = "Clinton"
				tsc = "Positive"
				toc = "ProClinton"
			if(loc[i]=="negative"):
				tcc = "Clinton"
				tsc = "Negative"
				toc = "AntiClinton"
				nnc += 1
			if(loc[i]=="neutral"):
				tcc = "Clinton"
				tsc = "Neutral"
				toc = "Neutral"
				nNc += 1
	if(tht==False):
		ngt  += 1
		tcc = "Neutral"
		tsc = "Neutral"
		toc = "Neutral"
	tht = False


	#############Information required to classify each tweet and its sentimentality to its corresponding bot:#########################
	fsn="/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/Bot-"+dfn.iloc[i].userid+"-EN.csv"

	#Assign Values to our political Bot Dictionary defined above:
	tmp = [tcc, tsc, toc]	#Temporary List

	if(dfn.iloc[i].userid in pbd.keys()):
		if(tmp not in pbd[dfn.iloc[i].userid]):
			tvl = dfn.iloc[i].userid	#temporary value
			pbd[tvl]=pbd[tvl]+[tmp]
	else:
		pbd[dfn.iloc[i].userid] = [tmp]
	
	#Assign values to temporary dataset that will stream these values into the designated csv file.
	dfc.loc[i] = [dfn.iloc[i].tweetid, dfn.iloc[i].userid, tcc, tsc]
	dfc[["tweetid", "userid","tweet_candidate_class", "tweet_sentiment_class"]].to_csv(fsn, mode='a', sep=',', header=False, index=False)

	#Clear this temporary dataset for it to be useable in the next iteration.
	dfc = dfc.iloc[i:]
	

#Printing our classification results:
print("******************Trump Sentimentality amongst bots:*******************")
print("Total no. of positive Trump tweets = " + str(npt))
print("Total no. of negative Trump tweets = " + str(nnt))
print("Total no. of neutral Trump tweets = " + str(nNt))
print("Total no. of Trump tweets = "+ str(npt+nnt+nNt))

print("******************Clinton Sentimentality amongst bots:*****************")
print("Total no. of positive Clinton tweets = " + str(npc))
print("Total no. of negative Clinton tweets = " + str(nnc))
print("Total no. of neutral Clinton tweets = " + str(nNc))
print("Total no. of Clinton tweets = "+ str(npc+nnc+nNc))

print("******************General Sentimentality amongst bots:*****************")
print("Total no. of general [not candidate related] tweets = " + str(ngt))

print("*****************General demographics of the bots:*********************")
nmc = 0	#Total No. of bots that represent multiple classes. I.e. Have multiple sentiments or are targetting multiple candidates.
npn = 0	#Total No. of bots that are both positive and negative in sentimentality.
ntc = 0	#Total No. of bots that target both Trump and Clinton.
nPtAc = 0	#Total No. of bots that are Pro Trump and Anti Clinton.
nPtAt = 0	#Total No. of bots that are Pro Trump and Anti Trump.
nAtPc = 0	#Total No. of bots that are Anti Trump and Pro Clinton.
nPcAc = 0	#Total No. of bots that are Pro Clinton and Anti Clinton.
nPtPc = 0	#Total No. of bots that are Pro Trump and Pro Clinton.
nAtAc = 0	#Total No. of bots that are Anti Trump and Anti Clinton.
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
print("Total no. of bots that have multiple classes = " +str(nmc))
print("Total no. of bots that are both positive and neagtive in sentimentality = " +str(npn))
print("Total no. of bots that target both Trump and Hillary = " +str(ntc))
print("Total no. of bots that are both ProTrump and AntiClinton = " +str(nPtAc))
print("Total no. of bots that are both ProTrump and AntiTrump = " +str(nPtAt))
print("Total no. of bots that are both AntiTrump and ProClinton = " +str(nAtPc))
print("Total no. of bots that are both ProClinton and AntiClinton = " +str(nPcAc))
print("Total no. of bots that are both ProTrump and ProClinton = " +str(nPtPc))
print("Total no. of bots that are both AntiTrump and AntiClinton = " +str(nAtAc))

'''
#Accuracy and Misclassification Rate of Classifier:
print("Accuracy Percentage of Classifier: " + str((tnr/len(loc))*100) + "%")
print("Misclassification Percentage of Classifier: " + str((1-(tnr/len(loc)))*100) + "%")
'''
