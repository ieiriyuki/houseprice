#!/usr/local/bin/pyton3
import pickle
import pandas as pd
import numpy as np
import sklearn.preprocessing as prp


df_org = pd.read_csv("./data/train.csv", header=0, sep=",", encoding='utf-8')

#with open('./pickles/df_org.pkl', 'wb') as opkl:
#    pickle.dump(df_org, opkl)

print("show head of df_org")
print(df_org.head(2))
print("shape of df_org {0}".format(df_org.shape))
print("columns of df_org are : {0}".format(list(df_org.columns)))

df_samp = df_org[['SalePrice', 'YrSold', 'MoSold', 'LotArea','BedroomAbvGr']]

print(df_samp.dtypes)
'''
for col in list(df_samp.columns):
    if any(list(df_samp[col].isna())):
        print("is col na? ", list(df_samp[col].isna()))
    else:
        print("not na")
'''

df_stat = pd.DataFrame({'mean': df_samp.mean(axis=0),
                        'vars': df_samp.var(axis=0)})
print("mean and vars are \n{0}".format(df_stat.T))

y_data = np.array([list(df_samp['SalePrice'])])

x_data = df_samp.iloc[:,1:]

scaler = prp.StandardScaler()
scaler.fit(x_data)

x_scal = scaler.transform(x_data)

df_stat = pd.DataFrame({'mean': x_scal.mean(axis=0),
                        'vars': x_scal.var(axis=0)})
print("mean and vars are \n{0}".format(df_stat.T))

nd_dump = np.concatenate((y_data.T, x_scal), axis=1)

with open('./pickles/nd_dump.pkl', 'wb') as opkl:
    pickle.dump(nd_dump, opkl)


#end of file
