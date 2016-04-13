# coding: utf-8
__author__ = 'roman'

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import xgboost as xgb
import random
import statistics
import pickle
import os
import math

def diffValid(a, b):
    if len(a) != len(b):
        print("Len error!")
    total = 0.0
    for i in range(len(a)):
        total += abs(a[i]-b[i])
    return total/len(a)


def diffValid_xg(yhat, y):
    y = y.get_label()
    return "error", diffValid(y,yhat)


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('Id')
    # output = list(train.columns.difference(['Id']))
    # output.remove('label_0')
    return output


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def findErrorValue(act, pred):
    if len(act) != len(pred):
        print('Length error!')
        exit()

    correct = 0
    for i in range(len(act)):
        if pred[i] > 0.5 and act[i] == 1:
            correct += 1
        if pred[i] <= 0.5 and act[i] == 0:
            correct += 1

    # https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation
    mean = correct/len(act)
    stdev = math.sqrt(mean*(1-mean)/len(act))

    return mean, stdev


def get_final_prediction(test, pred):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total = 0
    for id in test['Id']:
        for i in range(0,9):
            if pred[i][total] > 0.5:
                if test['label_{}'.format(i)].iloc[total] == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if test['label_{}'.format(i)].iloc[total] == 1:
                    false_negative += 1
        total += 1
    p = true_positive/(true_positive + false_positive)
    r = true_positive/(true_positive + false_negative)
    score = 2*p*r/(p+r)
    return score


def run_cross_validation(train, test, features, target, nfolds=20, random_state=0):
    eta = random.uniform(0.02, 0.3)
    max_depth = random.randint(2, 12)
    subsample = random.uniform(0.5, 0.95)
    colsample_bytree = random.uniform(0.5, 0.95)

    if 0:
        eta = 0.2
        max_depth = 2
        subsample = 0.5
        colsample_bytree = 0.5

    params = {
        "objective": "multi:softmax",
        "num_class": 2,
        # "eval_metric": "error",
        # "eval_metric": "rmse",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "nthread": 6,
        "seed": random_state
    }
    num_boost_round = 10000
    early_stopping_rounds = 100

    yfull_train = dict()
    yfull_test = []
    kf = KFold(len(train), n_folds=nfolds, shuffle=True, random_state=random_state)
    for train_index, test_index in kf:
        X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]
        y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[test_index]

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

        yhat = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_ntree_limit)

        # Each time store portion of precicted data in train predicted values
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = yhat[i]

        test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit)
        yfull_test.append(test_prediction)

    # Copy dict to list
    train_res = []
    for i in sorted(yfull_train.keys()):
        train_res.append(yfull_train[i])

    # Find median for KFolds on test
    valid_res = []
    for j in range(len(yfull_test[0])):
        all = []
        for i in range(nfolds):
            all.append(float(yfull_test[i][j]))
        median = statistics.median(all)
        # print(all, median)
        valid_res.append(median)

    return train_res, valid_res


def read_test_train(i):

    print("Load TRAIN..." + str(i) + "...")
    train = pd.read_csv("../modified_data_inception21/train_" + str(i) + ".csv")
    train.fillna(0, inplace=True)
    append = ["../modified_data/train_" + str(i) + ".csv", "../modified_data_inceptionBN/train_" + str(i) + ".csv"]
    for a in append:
        train1 = pd.read_csv(a)
        train1.fillna(0, inplace=True)
        print("Train features: {}".format(len(train.columns.values)))
        output = list(train.columns.values)
        output1 = list(train1.columns.values)
        output = intersect(output, output1)
        onlyuse = list(train1.columns.difference(output))
        print("Append features: {}".format(len(onlyuse)))
        # print(onlyuse)
        train = pd.concat([train, train1[onlyuse]], axis=1)
        print("Overall features: {}".format(len(train.columns.values)))

    # check_multi(train)

    print("Load TEST..." + str(i) + "...")
    test = pd.read_csv("../modified_data_inception21/test_" + str(i) + ".csv")
    test.fillna(0, inplace=True)
    append = ["../modified_data/test_" + str(i) + ".csv", "../modified_data_inceptionBN/test_" + str(i) + ".csv"]
    for a in append:
        test1 = pd.read_csv(a)
        test1.fillna(0, inplace=True)
        output = list(test.columns.values)
        output1 = list(test1.columns.values)
        output = intersect(output, output1)
        onlyuse = list(test1.columns.difference(output))
        # print(onlyuse)
        test = pd.concat([test, test1[onlyuse]], axis=1)
        print("Overall features: {}".format(len(test.columns.values)))

    features = get_features(train, test)
    print("Final features length: {}".format(len(features)))
    return train, test, features


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = './subm/submission_' + str(score) + str(now.strftime("%Y-%m-%d-%H-%M"))+ '.csv'
    f = open(sub_file, 'w')
    total = 0
    f.write('business_id,labels\n')
    for id in test['Id']:
        str1 = str(id) + ','
        for i in range(0, 9):
            if (prediction[i][total] > 0.5):
                str1 += str(i) + ' '
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_predictions_for_all_targets(train, test, features, nfolds=20, random_state=0):
    prediction = dict()
    correct = dict()
    valid_target = dict()
    for i in range(0, 9):
        target = 'label_{}'.format(i)
        valid_target[i], prediction[i] = run_cross_validation(train, test, features, target, nfolds, random_state)
        correct[i], stdev = findErrorValue(train[target].values, valid_target[i])
        print('Correct values for label {}: {:.6f} [Range: {:.6f} - {:.6f}]'.format(i, correct[i], correct[i]-3*stdev, correct[i]+3*stdev))

    avg_correct = 0
    for i in range(0, 9):
        avg_correct += correct[i]
    avg_correct /= 9
    print('Average correct value: {}'.format(avg_correct))

    score = get_final_prediction(train, valid_target)
    print('Predicted score = {}'.format(score))

    return valid_target, prediction


# Extend train and test with predicted values for more precize prediction
def extend_with_calculated_data(train, test, features, valid_target, prediction):
    train_2 = train.copy()
    test_2 = test.copy()
    features_2 = features.copy()
    for i in range(0, 9):
        new_feature = 'label_' + str(i) + '_predicted'
        train_2.loc[:, new_feature] = pd.Series(valid_target[i], index=train_2.index)
        test_2.loc[:, new_feature] = pd.Series(prediction[i], index=test_2.index)
        features_2.append(new_feature)

    return train_2, test_2, features_2


def merge_many_predictions(predictions):
    pred = dict()
    for i in range(len(predictions[0])):
        pred[i] = []
        for j in range(len(predictions[0][0])):
            lst = []
            for el in predictions:
                lst.append(predictions[el][i][j])
            pred[i].append(statistics.median(lst))

    return pred


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def read_all_cache_data():
    valid_target_path = os.path.join('cache', 'valid_target.txt')
    prediction_path = os.path.join('cache', 'prediction.txt')
    valid_target1 = restore_data(valid_target_path)
    prediction1 = restore_data(prediction_path)

    valid_target_path = os.path.join('cache1', 'valid_target.txt')
    prediction_path = os.path.join('cache1', 'prediction.txt')
    valid_target2 = restore_data(valid_target_path)
    prediction2 = restore_data(prediction_path)

    total = 0
    valid_target_all = dict()
    prediction_all = dict()
    for i in valid_target1.keys():
        valid_target_all[total] = valid_target1[i]
        prediction_all[total] = prediction1[i]
        total += 1
    for i in valid_target2.keys():
        valid_target_all[total] = valid_target2[i]
        prediction_all[total] = prediction2[i]
        total += 1

valid_target_path = os.path.join('cache', 'valid_target.txt')
prediction_path = os.path.join('cache', 'prediction.txt')
valid_target = restore_data(valid_target_path)
prediction = restore_data(prediction_path)
for i in range(52):
    random.seed(i)
    if i in valid_target.keys() and i in prediction.keys():
        print('Test {} already calculated. Skip it!'.format(i))
        continue
    train, test, features = read_test_train(i%20+1)
    # train, test, features = read_test_train(10)
    valid_target[i], prediction[i] = get_predictions_for_all_targets(train, test, features, random.randint(5, 20), i)
    train_2, test_2, features_2 = extend_with_calculated_data(train, test, features, valid_target[i], prediction[i])
    valid_target[i], prediction[i] = get_predictions_for_all_targets(train_2, test_2, features_2, random.randint(5, 20), i)
    cache_data(valid_target, valid_target_path)
    cache_data(prediction, prediction_path)


single_valid_target = merge_many_predictions(valid_target)
single_prediction = merge_many_predictions(prediction)

train, test, features = read_test_train(1)
score = get_final_prediction(train, single_valid_target)
print('Real score = {}'.format(score))
create_submission(score, test, single_prediction)

# 0.80
# Predicted score = 0.8113329799140013 (KFOld = 10)
# Predicted score = 0.8103104862331576 (Mean instead of median)
# Predicted score = 0.8207115181401902 (With second stage)
# Predicted score = 0.8352306067362986 (KFOld = 20)
# Predicted score = 0.8547693395392683 (KFOld = 40)
