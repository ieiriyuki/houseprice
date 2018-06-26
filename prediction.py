#!/usr/local/bin/python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df_test = pd.read_csv("./data/test.csv", header=0, sep=",", encoding='utf-8')
