#!/usr/local/bin/python
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as prp

df_test = pd.read_csv("./data/test.csv", header=0, sep=",", encoding='utf-8')

print(df_test.head(3))
print("shape of df_test: {0}".format(df_test.shape))

nd_test = np.array(df_test[['YrSold', 'MoSold', 'LotArea','BedroomAbvGr']])

print("shape of nd_test: {0}".format(nd_test.shape))
print(nd_test[:3])

with open('./pickles/mdl_smp_dict.pkl','rb') as ipkl:
    mdl_smp_dict = pickle.load(ipkl)

print("mdl_smp_dict keys are : {0}".format(mdl_smp_dict.keys()))

lm = mdl_smp_dict['model']
scaler = mdl_smp_dict['scale']

nd_trns = scaler.transform(nd_test)
sale_price = lm.predict(nd_trns).flatten().tolist()

submit = pd.DataFrame({'Id': df_test['Id'],
                        'SalePrice': sale_price})

print(submit.head(3))

submit.to_csv('./data/trace_submission.csv', index=False)

# end of file
