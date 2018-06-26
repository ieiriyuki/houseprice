#!/usr/local/bin/python
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mtr

with open('./pickles/stor_dict.pkl', 'rb') as ipkl:
    stor_dict = pickle.load(ipkl)

nd_dump = stor_dict['nd']

print("shape of nd_dump: {0}".format(nd_dump.shape))

y_data, x_data = np.split(nd_dump, [1], axis=1)

print("y shape", y_data.shape)
print("x shape", x_data.shape)

x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3)

lm = LinearRegression()
lm.fit(x_train, y_train)

print("coef: ", lm.coef_)
print("intc: ", lm.intercept_)

rmse_train = np.sqrt(mtr.mean_squared_error(y_train,
                lm.predict(x_train)))
print("rmse_train: {0}".format(np.log10(rmse_train)))
rmse_test = np.sqrt(mtr.mean_squared_error(y_test,
                lm.predict(x_test)))
print("rmse_test: {0}".format(np.log10(rmse_test)))

mdl_smp_dict = {'model': lm, 'scale': stor_dict['scale']}

with open('./pickles/mdl_smp_dict.pkl','wb') as opkl:
    pickle.dump(mdl_smp_dict, opkl)

# end of file
