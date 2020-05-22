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

	return text


def likelihoodFunctionInformation(txt, ldf):
	tsv  = 0	#Total Sentiment Value
	npw = 0		#No. of positive words
	nnw = 0		#No. negative words
	nNw = 0		#No. of neutral words

	psv = 0		#Previous Word sentiment value
	nac = False	#Negative conjuctive Adverb check
	wrd = " "	#Word to parse
	for ewt in txt.split():
		#Check for all versions of word in Sentiment Dictionary:
		print(ewt)
		sll = ObtainStemAndLemmatizationWord(ewt)       #Obtaining the noun version and root version of word using the function.
		print(sll)
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
				#print(sfw, 1, wsv, tsv)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif pfw.lower() in ldf[0].word.values:	#Check if plural version of word is in dataframe1
				wsv = int(ldf[0].iloc[ldf[0]['word'].loc[lambda x: x==pfw.lower()].index.tolist()[0]].sentiment)
				#print(pfw, 1, wsv, tsv)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif sfw.lower() in ldf[1].word.values:	#Check if singular version of word is in dataframe2
				#print(sfw, 2)
				wsv = int(ldf[1].iloc[ldf[1]['word'].loc[lambda x: x==sfw.lower()].index.tolist()[0]].sentiment)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif pfw.lower() in ldf[1].word.values:	#Check if plural version of word is in dataframe2
				#print(pfw, 2)
				wsv = int(ldf[1].iloc[ldf[1]['word'].loc[lambda x: x==pfw.lower()].index.tolist()[0]].sentiment)
				if(wsv>0):
					npw += 1
				elif(wsv<0):
					nnw += 1
				tsv += wsv
				psv = wsv
			elif sfw.lower() in ldf[2].word.values:	#Check if singular version of word is in dataframe3
				#print(sfw, 3, tsv)
				npw += 1
				psv = 3
			elif pfw.lower() in ldf[2].word.values:	#Check if plural version of word is in dataframe3
				#print(pfw, 3, tsv)
				npw += 1
				psv = 3
			elif sfw.lower() in ldf[3].word.values:	#Check if singular version of word is in dataframe4
				#print(sfw, 4)
				nnw += 1
				psv = -3
			elif pfw.lower() in ldf[3].word.values:	#Check if plural version of word is in dataframe4
				#print(pfw, 4)
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

	return(npw, nnw, nNw, tsv)

def NaiveBayes(txt, ppl, ldf):
	tov = likelihoodFunctionInformation(ctt, [df1, df2, df3, df4])	#Obtain tuple of values required to calculate the Likelihood funnction and posterior probability
	pPp = ppl[0]	#Positive class Prior Probability
	pnp = ppl[1]	#Negative class Prior Probability
	pNp = ppl[2]	#Neutral class Prior Probability
	npw = tov[0]	#No. of positive words
	nnw = tov[1]	#No. of negative words
	nNw = tov[2]	#No. of neutral words
	tsv = tov[3]	#Total Sentiment Value
	tnw = npw + nnw + nNw	#Total no. of words
	cls = " "	#Defining the class which the text belongs to.

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
			den = (pPp*(1-np.exp(-1*((npw*5*tsv)/(tnw))))) + (pnp*(1-np.exp(-1*((nnw*5)/(tnw))))) + (pNp*(1-np.exp(-1*((nNw)/(tnw)))))        #Calculate the denominator for the posterior probabilities.

			#Posterior Probability of sentiment of text is positive given the text:
			ppp = (pPp*(1-np.exp(-1*((npw*5*tsv)/(tnw)))))/(den)
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
		else:
			den = (pPp*(1-np.exp(-1*((npw*5)/(tnw))))) + (pnp*(1-np.exp(-1*((nnw*5*abs(tsv))/(tnw))))) + (pNp*(1-np.exp(-1*((nNw)/(tnw)))))        #Calculate the denominator for the posterior probabilities.

			#Posterior Probability of sentiment of text is positive given the text:
			ppp = (pPp*(1-np.exp(-1*((npw*5*tsv)/(tnw)))))/(den)
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
	return cls

#############Loading the Datasets:####################
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Training Dataset:
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/testingdataset.csv", sep=",", skiprows=[0], header=None, usecols=[0,1], names=["tweet_text","sentiment"])

#Sentiment Dataset 1:
df1 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/AFINN-111.txt", sep="\t", header=None, usecols=[0,1], names=["word","sentiment"])

#Sentiment Dataset 2:
df2 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/AFINN-96.txt", sep="\t", header=None, usecols=[0,1], names=["word","sentiment"])

#Sentiment Dataset 3 [Positive Words Only]:
df3 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/Positivewords.txt", sep="\n", header=None, usecols=[0], names=["word"])

#Sentiment Dataset 4 [Negative Words Only]:
df4 = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/SentimentDictionary/Negativewords.txt", sep="\n", header=None, usecols=[0], names=["word"])


#############Running the Naive Bayesian Classifer:####################

#Obtain the list of Prior Probabilities:
ppl = priorProb(dfn.sentiment)

loc = []	#List of classes for each text row in the dataframe.
for index, row in dfn.iterrows():
	#print(CleanUp(expandContractions(row["tweet_text"].replace("’", "'"))))
	ctt = CleanUp(expandContractions(row["tweet_text"].replace("’", "'")))	#Cleaned Tweet Text
	cot = NaiveBayes(ctt, ppl, likelihoodFunctionInformation(ctt, [df1, df2, df3, df4]))
	#print(cot)
	loc.append(cot)

tnr = 0	#Total No. of right words.
mcp = 0	#MisClassification percentage.
tap = 0	#Total Accuracy percentage.
for i in range(0,len(loc)):
	print(dfn.iloc[i].tweet_text)
	print("Actual Sentiment: " + dfn.iloc[i].sentiment)
	print("Calculated Sentiment: " + loc[i])
	if(loc[i].lower()==dfn.iloc[i].sentiment.lower()):
		tnr += 1
print("Accuracy Percentage of Classifier: " + str((tnr/len(loc))*100) + "%")
print("Misclassification Percentage of Classifier: " + str((1-(tnr/len(loc)))*100) + "%")
