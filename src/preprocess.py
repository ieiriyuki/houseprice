#!/usr/local/bin/pyton3

import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


DEFAULT_INPUT = 'data/raw/train.csv'
DEFAULT_OUTPUT = 'data/pickles/dataset.pkl'
DESCRIPTION = 'load data, '\
              'extract features, and'\
              'split to train & validation'
RANDOM_SEED = 1234
TEST_SIZE = 0.25


def main():
    parser=parse()

    raw_data = read_data(DEFAULT_INPUT)
    features = ['SalePrice', 'YrSold', 'MoSold', 'LotArea','BedroomAbvGr']
    extracted_data = raw_data[features].values

    x_train, x_test, y_train, y_test = train_test_split(extracted_data[:,1:4],
                                                        extracted_data[:,0],
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_SEED)

    x_scaler, x_train = scaling(x_train)
    y_scaler, y_train = scaling(y_train.reshape(-1, 1))

    _, x_test = scaling(x_test, x_scaler)
    _, y_test = scaling(y_test.reshape(-1, 1), y_scaler)

    dataset = { 'x_train': x_train,
                'x_test': x_test,
                'y_train': y_train,
                'y_test': y_test,
                'x_scale': x_scaler,
                'y_scale': y_scaler }

    with open(DEFAULT_OUTPUT, 'wb') as opkl:
        pickle.dump(dataset, opkl)


def read_data(input):
    data = pd.read_csv(input,
                       header=0,
                       sep=",",
                       encoding='utf-8')
    return data


def parse():
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     prog='preprocess')
    parser.add_argument('-i',
                        '--input',
                        default=DEFAULT_INPUT,
                        help='specify an input file',
                        metavar='input')
    parser.add_argument('-o',
                        '--output',
                        default=DEFAULT_OUTPUT,
                        help='specify an output file',
                        metavar='output')
    parser.add_argument('-s',
                        '--seed',
                        default=RANDOM_SEED,
                        help='specify the initial random seed',
                        metavar='seed')
    parser.add_argument('-t',
                        '--test-size',
                        default=TEST_SIZE,
                        help='specify the size of test data set',
                        metavar='t')
    return parser


def scaling(data, scaler=None):
    if scaler==None:
        scaler = StandardScaler()
    else :
        scaler=scaler
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data


if __name__=="__main__":
    main()
