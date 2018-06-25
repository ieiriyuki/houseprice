#!/usr/local/bin/pyton3
import pickle
import pandas as pd
import sklearn.preprocessing as prp


df_org = pd.read_csv("./data/train.csv", header=0, sep=",", encoding='utf-8')

print("show head of df_org")
print(df_org.head())

print("shape of df_org {0}".format(df_org.shape))

print("columns of df_org are : {0}".format(df_org.columns))

with open('./pickles/df_org.pkl', 'wb') as opkl:
    pickle.dump(df_org, opkl)
