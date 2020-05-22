import pandas as pd
import numpy as np
import csv

dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/datasets/CoreBotTweetsCombinedEN.csv", sep=",", skiprows=[0], header=None, usecols=[1], names=["userid"])
column_values = dfn[["userid"]].values.ravel()
unique_values =  pd.unique(column_values)
pd.DataFrame(unique_values).to_csv("ListIDS.csv", index=False)


'''
dfn = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/CoreBotEN/MachineLearning/NaiveBayes/ListIDS.csv", sep="\n", skiprows=[0], header=None, usecols=[1], names=["userid"], chunksize=2000)
dfi = pd.read_csv("/root/.encrypted/.pythonSai/kCoreBots/kCoreBotsList.csv", sep=",", header=None, chunksize=2000)
df_lst = pd.DataFrame(columns=["userid"])
kcu=[]

for df_ in dfi:
        #t0 = time.time()
        for i in df_.values:
                kcu.append(i[0])
for i in kcu:
        for df_ in dfn:
                df_lst = df_.loc[df_["userid"].map(lambda x: x==i)]

df_lst.to_csv("ListCommonIDS.csv", index=False)

'''
