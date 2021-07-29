import numpy as np
import os
import time
import json
import pandas as pd
from sklearn import metrics
from sklearn import decomposition
from sklearn.metrics import mean_squared_error,classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle

encode = False
y_le = LabelEncoder() 

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
    '''
    This function fetches the file from the compute job 
    metadata and returns the X and Y(target) dataframes 
    as numpy arrays.
    '''
    print('Starting compute job with the following input information:')
    print(json.dumps(job_details, sort_keys=True, indent=4))

    first_did = job_details['dids'][0]
    filename = job_details['files'][first_did][0]
    # filename = job_details # remove this later
    global encode
    global y_le

    df = pd.read_csv(filename)

    X = df.drop(["target"], axis = 1)
    Y = df['target']

    str_cols = [col for col in X.columns if hasattr(X[col], 'str')]
    for categ in str_cols:
        le = LabelEncoder()
        X[categ] = le.fit_transform(X[categ])

    if hasattr(Y, 'str'):
        Y = y_le.fit_transform(Y)
        encode = True
    
    X = X.to_numpy()
    if not encode:
        Y = Y.to_numpy()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    return (X_train, X_test, Y_train, Y_test)

def decode_results(encode, Y):
    global y_le
    if encode:
        Y = list(y_le.inverse_transform(Y))

    return(Y)

class Supervised:
    def __init__(self, algo, X_train,X_test,Y_train,Y_test):
        sc = StandardScaler()
        self.algo = algo
        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.transform(X_test)
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.model = None
        self.predictions = None
        self.accuracy = None
        self.confusion_matrix = None
        self.classification_report = None

        if(algo == 'decision_tree'):
            self.model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

        elif(algo == 'mlp_classifier'):
            self.model = MLPClassifier(hidden_layer_sizes=(50), activation='tanh', alpha =0.009, max_iter=250 )

        elif(algo == 'naive_bayes'):
            self.model = GaussianNB()

        elif(algo == 'knn'):
            self.model = KNeighborsClassifier(n_neighbors=3)

        elif(algo == 'linear_regression'):
            self.model = LinearRegression()

        elif(algo == 'logistic_regression'):
            self.model = LogisticRegression(C=1e5)
        
        elif(algo == 'MLP_Regressor'):
            self.model = MLPRegressor(random_state = 1, max_iter = 500)

        elif(algo == 'svm_classification'):
            self.model = svm.SVC(kernel='linear')

        elif(algo == 'random_forest_regressor'):
            self.model = RandomForestRegressor(n_estimators=100, max_depth=2)
        
        elif(algo == 'svr'):
            self.model = SVR(kernel='linear')
    
        elif(algo == 'elastic_net'):
            self.model = ElasticNet(alpha=0.01)

        elif(algo == 'non_lin_svm'):
            self.model = svm.NuSVC(gamma = 'auto')

        elif(algo == 'cart'):
            self.model = DecisionTreeClassifier(random_state = 100)


    def train(self):
        if self.algo == 'random_forest_regressor':
            self.X_train = self.X_train.values.reshape(-1,1)
            self.X_test = self.X_test.values.reshape(-1,1)

        self.model.fit(self.X_train, self.Y_train)

    def predict(self):
        self.predictions = self.model.predict(self.X_test)
        self.accuracy = self.model.score(self.X_test, self.Y_test)
        self.confusion_matrix = confusion_matrix(self.Y_test, self.predictions)
        self.classification_report = classification_report(self.Y_test, self.predictions)
        self.predictions = decode_results(encode, self.predictions)
        result = str(self.predictions) + '\n' + str(self.accuracy) + '\n' + str(self.confusion_matrix) + '\n' + str(self.classification_report)
        return result

class Unsupervised:
    def __init__(self, algo, X_train,X_test,Y_train,Y_test):
        self.model = None
        self.predictions = None
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        if(algo == 'k_means'):
            self.model = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        
        elif(algo == 'pca'):
            self.model = decomposition.PCA(n_components=self.X_train.shape[0])
        
        elif(algo == 't_sne'):
            self.model = TSNE(n_components=2, n_iter=1000, random_state=42)


    def predict(self):
        self.predictions = self.model.fit_transform(self.X_train)
        return self.predictions



if __name__ == '__main__':

    job = get_job_details()
    did = job['algo']['did']

    # job = 'iris.csv'
    # did = '2907E1f782f59C7B515c80B2DDB9DaC388F377F5'

    did_mapping = {'07A7287F45471dA8d7BddC647d49f03a54672E38': 'supervised/decision_tree', 
                   '2907E1f782f59C7B515c80B2DDB9DaC388F377F5': 'supervised/mlp_classifier', 
                   '2331e8116b0acB7AC164d8B4F332Aa104Eb9790F': 'supervised/naive_bayes',
                   '52D80495e56CFB241fD9e06aF7b5B96a80Ba509F': 'unsupervised/k_means',
                   '47838B1A397ed620F51D079Cd456d6564f940aC8': 'supervised/knn',
                   '3476E489Fb00058298fC8959Cb535fD3C29612c2': 'supervised/linear_regression',
                   '50374cfC875D9a66Be3f795Ff52Cb7714819eb7A': 'supervised/logistic_regression',
                   'e90F3344D508d017564b8EB4BB7e2E7C858365aa': 'supervised/MLP_Regressor',
                   '03115a5Dc5fC8Ff8DA0270E61F87EEB3ed2b3798': 'supervised/svm_classification',
                   '87F1A31A008D9cBE5c49B06dDb608df56967Cd51': 'supervised/random_forest_regressor',
                   '60b71c78E17de953A84f3A5A876645Cf15c1A92b': 'supervised/svr',
                   '0': 'supervised/elastic_net',
                   '1': 'unsupervised/t_sne',
                   '2': 'supervised/non_lin_svm',
                   '3': 'unsupervised/pca',
                   '4': 'supervised/cart'            
                   }

    algo_type = did_mapping[did].split('/')[0]
    algo = did_mapping[did].split('/')[1]

    X_train,X_test,Y_train,Y_test = preprocess_data(job)


    if algo_type == 'unsupervised':
        unsupervised = Unsupervised(algo, X_train, Y_train, X_test, Y_test)
        result =unsupervised.predict()
        print(result)
    elif algo_type == 'supervised':
        supervised = Supervised(algo, X_train,X_test,Y_train,Y_test)
        supervised.train()
        result = supervised.predict()
        print(result)