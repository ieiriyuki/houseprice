#!/usr/local/bin/python
import pickle
import sklearn.preprocessing as prp
from sklearn.linear_model import LinearRegression

with open('./pickles/df_org.pkl', 'rb') as ipkl:
    df_org = pickle.load(ipkl)

print("shape of df_org ?: {0}".format(df_org.shape))
