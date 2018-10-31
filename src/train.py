#!/usr/local/bin/python

import argparse, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as mtr


DEFAULT_DATASET = 'data/pickles/dataset.pkl'
DEFAULT_OUTPUT = 'data/pickles/model.pkl'
DESCRIPTION = 'load dataset, '\
              'train either model of '\
              'linear regression or random forest, '\
              'monitor metrics'


def main():
    parser = parse()
    args = parser.parse_args()

    with open(args.dataset, 'rb') as ipkl:
        dataset = pickle.load(ipkl)

    x_train, x_test, y_train, y_test = \
        dataset['x_train'], dataset['x_test'], dataset['y_train'], dataset['y_test']
    #print(x_test.shape)

    model = LinearRegression()
    model.fit(x_train, y_train)

    r2_train = model.score(x_train, y_train)
    r2_test = model.score(x_test, y_test)
    print('r2_train: {0:1.8f}, r2_test: {1:1.8f}'.format(r2_train, r2_test))

    y_train_pred, y_test_pred = model.predict(x_train), model.predict(x_test)

    y_scaler = dataset['y_scaler']
    y_train_resc, y_train_pred_resc, y_test_resc, y_test_pred_resc = \
        reverse_scale(y_train, y_scaler), \
        reverse_scale(y_train_pred, y_scaler), \
        reverse_scale(y_test, y_scaler), \
        reverse_scale(y_test_pred, y_scaler)

    rmsle_train = np.sqrt(mtr.mean_squared_log_error(y_train_resc,
                                                     y_train_pred_resc))
    rmsle_test = np.sqrt(mtr.mean_squared_log_error(y_test_resc,
                                                    y_test_pred_resc))
    print("rmsle_train: {0:1.8f}, rmsle_test: {1:1.8f}".format(rmsle_train,
                                                               rmsle_test))

    model_set = { 'features': dataset['features'],
                  'model': model,
                  'x_scaler': dataset['x_scaler'],
                  'y_scaler': dataset['y_scaler'] }

    with open(args.output, 'wb') as opkl:
        pickle.dump(model_set, opkl)


def reverse_scale(data, scaler):
    mean = scaler.mean_
    sd = np.sqrt(scaler.var_)
    data = (data * sd) + mean
    return data


def parse():
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     prog='training')
    parser.add_argument('--dataset',
                        default=DEFAULT_DATASET,
                        help='specify an dataset',
                        metavar='data')
    parser.add_argument('-o',
                        '--output',
                        default=DEFAULT_OUTPUT,
                        help='specify an output file',
                        metavar='output')
    parser.add_argument('--model',
                        choices=['lm', 'rf'],
                        default='lm',
                        help='specify a model trained')
    return parser


if __name__=='__main__':
    main()
