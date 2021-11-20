import os
from pprint import pprint

import pandas as pd
from joblib import dump, load
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from data_parser import createDataset, tfidf_transform
import argparse
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def getArgparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='A3-files')
    parser.add_argument('--save_dir', type=str, default='saved')

    return parser

def train_SVC(df, features_train, labels_train, features_test, labels_test):
    C = [.0001, .001, .01, 1]
    degree = [1, 2, 3, 4, 5]
    gamma = [.0001, .001, .01, .1, 1, 10, 100]
    kernel = ['linear', 'rbf', 'poly']
    probability = [True]
    # Create the random grid
    random_grid = {'C': C,
                'kernel': kernel,
                'gamma': gamma,
                'degree': degree,
                'probability': probability
                }
    # pprint(random_grid)

    svc = svm.SVC(random_state=42)
    random_search = RandomizedSearchCV(estimator=svc,
                                    param_distributions=random_grid,
                                    n_iter=50,
                                    scoring='accuracy',
                                    cv=3,
                                    verbose=1,
                                    random_state=42)
    
    random_search.fit(features_train, labels_train)
    best_svc = random_search.best_estimator_
    best_svc.fit(features_train, labels_train)
    
    return best_svc



def conf_matrix(df, labels_test, svc_pred, setting):
    # Confusion matrix
    aux_df = df[['Label', 'label_code']].drop_duplicates().sort_values('label_code')
    conf_matrix = confusion_matrix(labels_test, svc_pred)

    plt.figure(figsize=(12.8,6))
    sns.heatmap(conf_matrix,
                annot=True,
                xticklabels=aux_df['Label'].values,
                yticklabels=aux_df['Label'].values,
                cmap="Blues")

    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix --- {}'.format(setting))
    # plt.savefig('{}.png'.format(aux_df['Label'].values[1]))
    plt.savefig('{}.png'.format(setting))

def train(args):
    df = createDataset(args.data_dir)

    setting = 'BOW'
    print(setting)
    features_train, labels_train, features_test, labels_test = tfidf_transform(df)
    svc = train_SVC(df, features_train, labels_train, features_test, labels_test)
    svc_pred = svc.predict(features_test)
    print(classification_report(labels_test,svc_pred))
    conf_matrix(df, labels_test, svc_pred, setting)
    
    setting = 'BOW + lemma'
    print(setting)
    features_train, labels_train, features_test, labels_test = tfidf_transform(df, text='lemma')
    svc = train_SVC(df, features_train, labels_train, features_test, labels_test)
    svc_pred = svc.predict(features_test)
    print(classification_report(labels_test,svc_pred))
    conf_matrix(df, labels_test, svc_pred, setting)

    setting = 'BOW + lemma - stop'
    print(setting)
    features_train, labels_train, features_test, labels_test = tfidf_transform(df, text='lemma - stop')
    svc = train_SVC(df, features_train, labels_train, features_test, labels_test)
    svc_pred = svc.predict(features_test)
    print(classification_report(labels_test,svc_pred))
    conf_matrix(df, labels_test, svc_pred, setting)

    setting = 'BOW + title + review + lemma - stop'
    print(setting)
    features_train, labels_train, features_test, labels_test = tfidf_transform(df, text='title + review + lemma - stop')
    svc = train_SVC(df, features_train, labels_train, features_test, labels_test)
    svc_pred = svc.predict(features_test)
    print(classification_report(labels_test,svc_pred))
    conf_matrix(df, labels_test, svc_pred, setting)

    setting = 'BOW + title + review + lemma'
    print(setting)
    features_train, labels_train, features_test, labels_test = tfidf_transform(df, text='title + review + lemma')
    svc = train_SVC(df, features_train, labels_train, features_test, labels_test)
    svc_pred = svc.predict(features_test)
    print(classification_report(labels_test,svc_pred))
    conf_matrix(df, labels_test, svc_pred, setting)

    # setting = 'BOW + title + lemma'
    # print(setting)
    # features_train, labels_train, features_test, labels_test = tfidf_transform(df, text='title + lemma')
    # svc = train_SVC(df, features_train, labels_train, features_test, labels_test)
    # svc_pred = svc.predict(features_test)
    # print(classification_report(labels_test,svc_pred))
    # conf_matrix(df, labels_test, svc_pred, setting)

    # setting = 'BOW + lemma - stop + senti'
    # print(setting)
    # features_train, labels_train, features_test, labels_test = tfidf_transform(df, text='lemma - stop', sentiscore=True)
    # svc = train_SVC(df, features_train, labels_train, features_test, labels_test)
    # svc_pred = svc.predict(features_test)
    # print(classification_report(labels_test,svc_pred))
    # conf_matrix(df, labels_test, svc_pred, setting)

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    # dump(svc, '{}/models.joblib'.format(args.save_dir))
    # dump([features_test, labels_test], '{}/test_data.joblib'.format(args.save_dir))


def test(args):
    cls = load(args.model_path)
    features_test, labels_test = load(args.test_data_path)
    svc_pred = cls.predict(features_test)
    print(classification_report(labels_test,svc_pred))

if __name__ == '__main__':
    argparser = getArgparser() 
    args = argparser.parse_args()
    args.model_path = '{}/models.joblib'.format(args.save_dir)
    args.test_data_path = '{}/test_data.joblib'.format(args.save_dir)
    
    if not os.path.exists(args.model_path):
        train(args)

    # test(args) 

    

