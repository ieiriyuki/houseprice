#!/usr/local/bin/python
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mtr

with open('./pickles/nd_dump.pkl', 'rb') as ipkl:
    nd_dump = pickle.load(ipkl)

print("shape of df_samp: {0}".format(nd_dump.shape))

y_data, x_data = np.split(nd_dump, [1], axis=1)

print("y shape", y_data.shape)
print("x shape", x_data.shape)

x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3)

lm = LinearRegression()
lm.fit(x_train, y_train)

print("coef: ", lm.coef_)
print("intc: ", lm.intercept_)

mse_train = np.sqrt(mtr.mean_squared_error(y_train,
                lm.predict(x_train)))
print("mse_train: {0}".format(np.log10(mse_train)))
mse_test = np.sqrt(mtr.mean_squared_error(y_test,
                lm.predict(x_test)))
print("mse_test: {0}".format(np.log10(mse_test)))

# end of file
