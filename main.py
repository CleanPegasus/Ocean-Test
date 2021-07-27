#Decision Tree Classification
import numpy as np
import os
import time
import json
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

def get_job_details():
    """Reads in metadata information about assets used by the algo"""
    job = dict()
    job['dids'] = json.loads(os.getenv('DIDS', None))
    job['metadata'] = dict()
    job['files'] = dict()
    job['algo'] = dict()
    job['secret'] = os.getenv('secret', None)
    algo_did = os.getenv('TRANSFORMATION_DID', None)
    if job['dids'] is not None:
        for did in job['dids']:
            # get the ddo from disk
            filename = '/data/ddos/' + did
            print(f'Reading json from {filename}')
            with open(filename) as json_file:
                ddo = json.load(json_file)
                # search for metadata service
                for service in ddo['service']:
                    if service['type'] == 'metadata':
                        job['files'][did] = list()
                        index = 0
                        for file in service['attributes']['main']['files']:
                            job['files'][did].append(
                                '/data/inputs/' + did + '/' + str(index))
                            index = index + 1
    if algo_did is not None:
        job['algo']['did'] = algo_did
        job['algo']['ddo_path'] = '/data/ddos/' + algo_did
    return job

def preprocess_data(job_details):
    print('Starting compute job with the following input information:')
    print(json.dumps(job_details, sort_keys=True, indent=4))

    first_did = job_details['dids'][0]
    filename = job_details['files'][first_did][0]

    df = pd.read_csv(filename)

    str_cols = [col for col in df.columns if hasattr(df[col], 'str')]
    for categ in str_cols:
        le = LabelEncoder()
        df[categ] = le.fit_transform(df[categ])

    X = df.drop(["target"], axis = 1)
    Y = df['target']

    return (X, Y)


def decision_tree(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    results = {'Test Prediction': y_pred, 'Confusion Matrix': cm, 'Accuracy': accuracy}
    
    f = open("/data/outputs/result.txt", "w")
    f.write(str(results))
    f.close()

def mlp_classifier(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    MLPobj = MLPClassifier(hidden_layer_sizes=(50), activation='tanh', alpha =0.009, max_iter=250 )
    MLPobj.fit(X_train,Y_train)

    predictions = MLPobj.predict(X_test)
    accuracy = MLPobj.score(X_test,Y_test)
    cm = confusion_matrix(Y_test,predictions)
    report = classification_report(Y_test,predictions)

    # now you can save it to a file
    with open('/data/outputs/filename.pkl', 'wb') as f:
        pickle.dump(MLPobj, f)
    f = open("/data/outputs/result.txt", "w")
    f.write('Predictions: '+str(predictions))
    f.write('Accuracy: '+str(accuracy))
    f.write('Confusion Matrix: \n' + str(cm))
    f.write('Classification Report: \n', + str(report))    
    f.close()

def naive_bayes(X, Y):
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    # Make predictions
    expected = y_test
    predicted = model.predict(X_test)
    # Evaluate the predictions
    accuracy = accuracy_score(expected, predicted)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(predicted))
    f.close()



if __name__ == '__main__':

    job = get_job_details()
    did = job['algo']['did']
    X, Y = preprocess_data(job)

    did_mapping = {'07A7287F45471dA8d7BddC647d49f03a54672E38': 'decision_tree', 
                   '2907E1f782f59C7B515c80B2DDB9DaC388F377F5': 'mlp_classifier', 
                   '2': 'naive_bayes'}

    algo = did_mapping[did]

    if(algo == 'decision_tree'):
        decision_tree(X, Y)
    elif(algo == 'mlp_classifier'):
        mlp_classifier(X,Y)
    elif(algo == 'naive_bayes'):
        naive_bayes(X, Y)



