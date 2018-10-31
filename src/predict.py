#!/usr/local/bin/python

import argparse, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from preprocess import read_data
from train import reverse_scale


DEFAULT_INPUT = 'data/raw/test.csv'
DEFAULT_OUTPUT = 'data/prepared/my_submission.csv'
DEFAULT_MODEL = 'data/pickles/model.pkl'
DESCRIPTION = 'load test data, '\
              'predict house price for test data, and'\
              'create a file for submission'


def main():
    parser = parse()
    args = parser.parse_args()

    raw_data = read_data(args.input)

    with open(args.model, 'rb') as ipkl:
        model_set = pickle.load(ipkl)

    features = model_set['features'][1:]
    test_data = raw_data[features]
    #print(test_data.shape)

    model, x_scaler, y_scaler = \
        model_set['model'], model_set['x_scaler'], model_set['y_scaler']

    scaled_test = x_scaler.transform(test_data)
    predicted_price = model.predict(scaled_test).flatten().tolist()
    predicted_price = reverse_scale(predicted_price, y_scaler)

    submit = pd.DataFrame({ 'Id': raw_data['Id'],
                            'SalePrice': predicted_price })

    with open(args.output, 'w') as ofile:
        submit.to_csv(ofile, index=False)


def parse():
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     prog='predict')
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
    parser.add_argument('--model',
                        default=DEFAULT_MODEL,
                        help='specify a model set for prediction'
                        metavar='model')
    return parser


if __name__=='__main__':
    main()
