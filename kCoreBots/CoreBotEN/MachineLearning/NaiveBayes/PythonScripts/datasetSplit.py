import csv
import pandas as pd
import random


##########Load the sample DataSet:#############
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/sampleDataset.csv", sep=",", skiprows=[0], header=None, usecols=[2,6], names=["tweet_text","sentiment"])
trn = pd.DataFrame(columns=["tweet_text", "sentiment"])
tst = pd.DataFrame(columns=["tweet_text", "sentiment"])

###########Create the Training DataSet (consisting of 1/3 of the data from the sample) using Randomized Holdout Cross Validation:#####################
trn=dfn.sample(24)
trn[["tweet_text", "sentiment"]].to_csv("trainingdataset.csv", mode='a', header=False, index=False)

###########Create the Testing DataSet (consisting of the remaining 2/3 of the data from the sample) using Randomized Holdout Cross Validation#########:
common = dfn.merge(trn,on=["tweet_text", "sentiment"])
tst = dfn[(~dfn.tweet_text.isin(common.tweet_text))]
tst[["tweet_text", "sentiment"]].to_csv("testingdataset.csv", mode='a', header=False, index=False)
