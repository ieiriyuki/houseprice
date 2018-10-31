#!/usr/local/bin/python

from sklearn.linear_model import LinearRegression
from sklearn.ensenmbe import RandomForestRegressor


class Model(object):
    def __init__(self, arg):
        # under construction
        # this may be abundant
        self.model = None

    def linearmodel(self):
        self.model = LinearRegression()
        return self.model

    def randomforest():
        self.model = RandomForestRegressor()
        return self.model


def main():
    print('print')


if __name__=='__main__':
    main()
