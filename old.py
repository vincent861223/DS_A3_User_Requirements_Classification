def testOneVOne(args):
    classifiers = load(args.model_path)
    test_data = load(args.test_data_path)
    
    labels_all = []
    pred_all = []
    i = 0
    for features_test, labels_test in test_data:
        labels_test = pd.DataFrame(labels_test).reset_index()
        # print(features_test)
        # print(labels_test)
        indexes = labels_test[labels_test['label_code'] == 1].index.to_numpy()
        labels_test = np.ones(indexes.shape) * i
        features_test = features_test[indexes, :]
        probs = []
        for cls in classifiers:
            cls_score = cls.predict_proba(features_test) 
            probs.append(cls_score[:,0])
            # print(cls_score[:,0])
        probs = np.stack(probs, axis=0)
        # print(probs.shape)
        pred = np.argmax(probs, axis=0) 
        print(labels_test)
        print(pred)
        # print(pred.shape)
        labels_all.append(labels_test)
        pred_all.append(pred)
        #print(classification_report(labels_test, pred))
        
        i += 1

    labels_all = np.concatenate(labels_all)
    pred_all = np.concatenate(pred_all)

    print(labels_all.shape)
    print(classification_report(labels_all, pred_all))


def trainOneVOVne(args):
    dfs = createDataset(args.data_dir)
    classifiers = []
    test_data = []
    for df in dfs:
        # print(df.loc[3]['Text'])
        # print(df.loc[3]['text_parsed_1'])
        # print(df.loc[3]['text_parsed_2'])
        # print(df.loc[3]['text_parsed_3'])
        # print(df.loc[3]['text_parsed_4'])
        svc, features_test, labels_test = train_SVM(df)
        test_data.append((features_test, labels_test))
        classifiers.append(svc)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dump(classifiers, '{}/models.joblib'.format(args.save_dir))
    dump(test_data, '{}/test_data.joblib'.format(args.save_dir))

def train_SVM1V1(df):
    features_train, labels_train, features_test, labels_test = tfidf_transform(df)
    # svc_0 =svm.SVC(random_state=42)
    # print('Parameters currently in use:\n')
    # pprint(svc_0.get_params()) 

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

    # print("The best hyperparameters from Random Search are:")
    # print(random_search.best_params_)
    # print("")
    # print("The mean accuracy of a model with these hyperparameters is:")
    # print(random_search.best_score_)

    # random search was better
    best_svc = random_search.best_estimator_

    # fit the model
    best_svc.fit(features_train, labels_train)
    svc_pred = best_svc.predict(features_test)
    # print(svc_pred)
    # print(best_svc.predict_proba(features_test))

    # # Training accuracy
    # print("The training accuracy is: ")
    # print(accuracy_score(labels_train, best_svc.predict(features_train)))

    # # Test accuracy
    # print("The test accuracy is: ")
    # print(accuracy_score(labels_test, svc_pred))
    
    # # Classification report
    # print("Classification report")
    print(classification_report(labels_test,svc_pred))

    conf_matrix(df, labels_test, svc_pred)
    return best_svc, features_test, labels_test