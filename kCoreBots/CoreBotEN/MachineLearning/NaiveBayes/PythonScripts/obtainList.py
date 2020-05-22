import pandas as pd
import numpy as np
import csv

#################Obtain the List of All bpts that sent Political Tweets [i.e. Political Bots]:#######################
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/CoreBotTweetsCombinedEN.csv", sep=",", skiprows=[0], header=None, usecols=[1], names=["userid"])
column_values = dfn[["userid"]].values.ravel()
unique_values =  pd.unique(column_values)
pd.DataFrame(unique_values).to_csv("ListIDS.csv", index=False)
