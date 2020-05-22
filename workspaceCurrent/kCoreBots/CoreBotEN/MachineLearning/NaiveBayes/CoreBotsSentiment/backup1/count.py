import pandas as pd
import numpy as np
import csv
import time


#############DataFrames:###################
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/KCoreTweetsCombinedRU.csv", sep=",", header=None, usecols=[1])
dfi = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/KCoreBotTweetsCombinedEN.csv", sep=",", header=None, usecols=[1])
new_df = ([pd.concat([dfn,dfi]).duplicated(subset=[1,1], keep=False)])
print(dfi[1].nunique())

'''
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/CoreBotsSentiment/network.csv", sep=",", header=None, usecols=[0,1,2,3])
dfn = dfn.drop_duplicates(subset=0, keep='first')
dfn[[0,1,2,3]].to_csv("network.csv", mode='w', header=False, index=False)
'''
