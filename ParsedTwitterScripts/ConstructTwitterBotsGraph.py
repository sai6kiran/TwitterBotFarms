import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

dfn = pd.read_csv("/root/.encrypted/.pythonSai/my_parsed.csv", sep=",", header=None, chunksize=2000, names=["tweetid","userid","is_retweet","retweet_userid","retweet_tweetid"])

G=nx.DiGraph()
tmp = 0
cnt = 0
for df_ in dfn:
        #t0 = time.time()
        tmp +=1
        print(len(df_))
        #for index, row in df_.iterrows():
                #cnt+=1
                #if(row["retweet_userid"] == ''):
                        #print(row, tmp, cnt)
                        #break
        df_lst = df_.loc[df_["retweet_userid"].map(lambda x: print(x, tmp) if x!='' else "false")]
        #t1 = time.time()
        print("************************************************************")
        #print(t1-t0)
        #if(row['retweet_userid'] not in uid and row['retweet_userid'] is not None):
                #G.add_node(row['retweet_userid'])
        #if(row['userid'] not in uid):
                #G.add_node(row['userid'])

