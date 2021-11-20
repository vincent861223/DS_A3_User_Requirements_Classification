import json
import glob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

def nltk_prepare():
    print("Downloading NLTK files...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    wordnet_lemmatizer = WordNetLemmatizer()

def createDataset(data_dir):
    data_files = glob.glob('./{}/*.json'.format(data_dir))
    for data_file in data_files:
        df = load_df(data_file)
        print(df.value_counts('Label'))
    return None

def load_df(file_path):
    data = load_file(file_path)
    df = pd.DataFrame(data, columns = ['Text', 'Label'])
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
    df = pd.DataFrame(data, columns = ['Text', 'Label'])
    return df

def convert_data(raw_data):
    data = []
    for elem in raw_data:
        data.append([elem["comment"], elem["label"]])

    return data


def replace_symbols(df):
    df['text_parsed_1'] = df['Text'].str.replace("\r", " ")
    df['text_parsed_1'] = df['text_parsed_1'].str.replace("\n", " ")
    df['text_parsed_1'] = df['text_parsed_1'].str.replace("  ", " ")
    df['text_parsed_1'] = df['text_parsed_1'].str.replace('"', '')
    df['text_parsed_1'] = df['text_parsed_1'].str.lower()

    punctuation_signs = list("?:!.,;")

    df['text_parsed_2'] = df['text_parsed_1']
    for punct_sign in punctuation_signs:
        df['text_parsed_2'] = df['text_parsed_2'].str.replace(punct_sign, '')

    df['text_parsed_2'] = df['text_parsed_2'].str.replace("'s", "")

    return df

def lemmatize(df):
    nrows = len(df)
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_text_list = []

    for row in range(0, nrows):
        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Save the text and its words into an object
        text = df.loc[row]['text_parsed_2']
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)

    df['text_parsed_3'] = lemmatized_text_list

    return df

def replace_stopwords(df):
    stop_words = list(stopwords.words('english'))
    df['text_parsed_4'] = df['text_parsed_3']

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['text_parsed_4'] = df['text_parsed_4'].str.replace(regex_stopword, '')

    return df

def replace_label(df):
    label_codes = {
        'Bug': 1,
        'Not_Bug': 0
    }

    # label mapping
    df['label_code'] = df['Label']
    df = df.replace({'label_code':label_codes})

    return df

def tfidf_transform(df):
     # Parameter election
    ngram_range = (1,2)
    min_df = 1
    max_df = 1.
    max_features = 300

    X_train, X_test, y_train, y_test = train_test_split(df['text_parsed'],
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
    print(features_train.shape)

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test
    print(features_test.shape)
    return features_train, labels_train, features_test, labels_test

def train_SVM(df):
    features_train, labels_train, features_test, labels_test = tfidf_transform(df)
    svc_0 =svm.SVC(random_state=42)
    print('Parameters currently in use:\n')
    pprint(svc_0.get_params()) 

    C = [.0001, .001, .01, -1, 1]
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
    pprint(random_grid)

    # First create the base model to tune
    svc = svm.SVC(random_state=42)
    
    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=svc,
                                    param_distributions=random_grid,
                                    n_iter=50,
                                    scoring='accuracy',
                                    cv=3,
                                    verbose=1,
                                    random_state=42)
    

    # Fit the random search model
    random_search.fit(features_train, labels_train)

    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

    # random search was better
    best_svc = random_search.best_estimator_

    # fit the model
    best_svc.fit(features_train, labels_train)
    svc_pred = best_svc.predict(features_test)
    print(svc_pred)
    print(best_svc.predict_proba(features_test))

    # Training accuracy
    print("The training accuracy is: ")
    print(accuracy_score(labels_train, best_svc.predict(features_train)))

    # Test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(labels_test, svc_pred))
    
    # Classification report
    print("Classification report")
    print(classification_report(labels_test,svc_pred))

    conf_matrix(df, labels_test, svc_pred)

def conf_matrix(df, labels_test, svc_pred):
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
    plt.title('Confusion matrix')
    plt.show()


if __name__ == '__main__':
    file_path = 'Bug_tt.json'
    nltk_prepare()
    df = load_df(file_path)
    df = replace_symbols(df)
    df = lemmatize(df)
    df = replace_stopwords(df)
    df = replace_label(df)

    print(df.loc[3]['Text'])
    print(df.loc[3]['text_parsed_1'])
    print(df.loc[3]['text_parsed_2'])
    print(df.loc[3]['text_parsed_3'])
    print(df.loc[3]['text_parsed_4'])

    list_columns = ["Text", "Label", "label_code", "text_parsed_4"]
    df = df[list_columns]
    df = df.rename(columns={'text_parsed_4': 'text_parsed'})

    
    train_SVM(df)

   

