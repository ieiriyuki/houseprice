import pandas as pd
import numpy as np
import sklearn.preprocessing as prp

def read_data(input):
    data = pd.read_csv(input,
                         header=0,
                         sep=",",
                         encoding='utf-8')
    return(data)

DEFAULT_INPUT = 'data/raw/train.csv'
df_org = read_data(DEFAULT_INPUT)

print(df_org.head(2))
print("shape of df_org {0}".format(df_org.shape))
print("columns of df_org are : {0}".format(list(df_org.columns)))

df_samp = df_org[['SalePrice', 'YrSold', 'MoSold', 'LotArea','BedroomAbvGr']]

df_stat = pd.DataFrame({'mean': df_samp.mean(axis=0),
                        'vars': df_samp.var(axis=0)})
print("mean and vars are \n{0}".format(df_stat.T))

y_data = np.array([list(df_samp['SalePrice'])])

x_data = df_samp.iloc[:,1:]

scaler = prp.StandardScaler()
scaler.fit(x_data)

x_scal = scaler.transform(x_data)

#print(x_scal.shape)

df_stat = pd.DataFrame({'mean': x_scal.mean(axis=0),
                        'vars': x_scal.var(axis=0)})
print("mean and vars are \n{0}".format(df_stat.T))
