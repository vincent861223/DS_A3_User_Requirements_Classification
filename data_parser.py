import json
import glob
import pandas as pd
from data_process import nltk_prepare, replace_stopwords, replace_symbols, lemmatize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

import seaborn as sns

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def createDataset(data_dir):
    nltk_prepare()
    dfs = []
    data_files = glob.glob('./{}/*.json'.format(data_dir))
    for data_file in data_files:
        df = load_df(data_file)
        df = df[df['Label'] == df['Label'].unique()[0]]
        # df = replace_symbols(df)
        
        # df = replace_stopwords(df)
        # df = replace_label(df)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    # df = replace_symbols(df)
    # df = lemmatize(df)
    # df = replace_stopwords(df)
    # df = replace_label(df)
    df['tmp'] = df['Text']
    df = replace_symbols(df)
    df = lemmatize(df)
    df = df.rename(columns={'tmp': 'lemma'})

    df['tmp'] = df['Text']
    df = replace_symbols(df)
    df = lemmatize(df)
    df = replace_stopwords(df)
    df = df.rename(columns={'tmp': 'lemma - stop'})
    df = replace_label(df)

    df['tmp'] = df['Text'].astype(str) + df['Title'].astype(str)
    df = replace_symbols(df)
    df = lemmatize(df)
    df = replace_stopwords(df)
    df = df.rename(columns={'tmp': 'title + review + lemma - stop'})
    df = replace_label(df)

    df['tmp'] = df['Text'].astype(str) + df['Title'].astype(str)
    df = replace_symbols(df)
    df = lemmatize(df)
    df = df.rename(columns={'tmp': 'title + review + lemma'})
    df = replace_label(df)

    # df['tmp'] = df['Title']
    # df = replace_symbols(df)
    # df = lemmatize(df)
    # df = df.rename(columns={'tmp': 'title + lemma'})
    # df = replace_label(df)
    return df

def load_file(file_path):
    """
    :param file_path: path to the json file
    :return: an array in which each entry is tuple [text, classification label]
    """
    with open(file_path) as json_file:
        raw_data = json.load(json_file)
        return convert_data(raw_data)

def load_df(file_path):
    data = load_file(file_path)
    df = pd.DataFrame(data, columns = ['Text', 'Title', 'SentiScore', 'Label'])
    return df

def convert_data(raw_data):
    data = []
    for elem in raw_data:
        data.append([elem["comment"], elem['title'], elem['sentiScore'], elem["label"]])

    return data


def replace_label(df):
    label_codes = { label: i for i, label in enumerate(df['Label'].unique())}

    # label mapping
    df['label_code'] = df['Label']
    df = df.replace({'label_code':label_codes})

    return df

def tfidf_transform(df, text='Text', bigram=True, sentiscore=False):
     # Parameter election
    ngram_range = (1,2)
    min_df = 1
    max_df = 1.
    max_features = 300 if not sentiscore else 299

    if sentiscore:
        X_train, X_test, senti_train, senti_test, y_train, y_test = train_test_split(df[text], 
                                                    df['SentiScore'],
                                                    df['label_code'],
                                                    test_size=0.15,
                                                    random_state=42) 
    else: 
        X_train, X_test, y_train, y_test = train_test_split(df[text],
                                                    df['label_code'],
                                                    test_size=0.15,
                                                    random_state=42)

    tfidf = TfidfVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            stop_words=None,
                            lowercase=False,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=max_features,
                            norm='l2',
                            sublinear_tf=True)

    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
   

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test

    if sentiscore:
        senti_train = np.expand_dims(senti_train, axis=1)
        features_train = np.concatenate((senti_train, features_train), axis=1)
        senti_test = np.expand_dims(senti_test, axis=1)
        features_test = np.concatenate((senti_test, features_test), axis=1)
        

    print(features_train.shape)
    # print(features_train[0])
    print(features_test.shape)

    # print(features_train, labels_train)
 
    return features_train, labels_train, features_test, labels_test



if __name__ == '__main__':
    df = createDataset('A3-files')
    print(df)
    print(df.loc[365]['Text'])
    print(df.loc[365]['lemma'])

    tfidf_transform(df, sentiscore=True)

    # data = load_vectors('wiki-news-300d-1M.vec')
    # print(data)

   

