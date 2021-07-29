import numpy as np
import os
import time
import json
import pandas as pd

encode = False
y_le = LabelEncoder() 

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

class Supervised:
    def __init__(self,X_train,X_test,Y_train,Y_test):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.transform(X_test)
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.model = None
        self.predictions = None
        self.accuracy = None
        self.confusion_matrix = None
        self.classification_report = None

    def decision_tree(self):
        self.model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

    def mlp_classifier(self):
        self.model = MLPClassifier(hidden_layer_sizes=(50), activation='tanh', alpha =0.009, max_iter=250 )

    def naive_bayes(self):
        self.model = GaussianNB()

    def linear_regression(self):
        self.model = LinearRegression()

    def logistic_regression(self):
        self.model = LogisticRegression(C=1e5)
    
    def MLP_Regressor(self):
        self.model = MLPRegressor(random_state = 1, max_iter = 500)

    def svm_classification(self):
        self.model = svm.SVC(kernel='linear')

    def random_forest_regressor(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=2)

    def svr(self):
        self.model = SVR(kernel='linear')

    def elasticnet(self):
        self.model = ElasticNet(alpha=0.01)

    def non_lin_svm(self):
        self.model = svm.NuSVC(gamma = 'auto')
    
    def CART(self):
        self.model= DecisionTreeClassifier(random_state = 100)
    
    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def predict(self):
        self.predictions = self.model.predict(self.X_test)
        self.accuracy = self.model.score(self.X_test, self.Y_test)
        self.confusion_matrix = confusion_matrix(self.Y_test, self.predictions)
        self.classification_report = classification_report(self.Y_test, self.predictions)
        result = str(self.predictions) + '\n' + str(self.accuracy) + '\n' + str(self.confusion_matrix) + '\n' + str(self.classification_report)
        return result
         
x,y, = preprocess_data()
sup = Supervised(X_train,X_test,Y_train,Y_test)
algo = input('Enter the algorithm to be used: ')
build_model(algo)


def build_model(algo):
    if(algo == 'decision_tree'):
        

        
    elif(algo == 'mlp_classifier'):
        mlp_classifier(X,Y)
    elif(algo == 'naive_bayes'):
        naive_bayes(X, Y)
    elif(algo == 'k_means'):
        k_means(X)
    elif(algo == 'knn'):
        knn(X, Y)
    elif(algo == 'linear_regression'):
        linear_regression(X, Y)
    elif(algo == 'logistic_regression'):
        logistic_regression(X, Y)
    elif(algo == 'MLP_Regressor'):
        MLP_Regressor(X, Y)
    elif(algo == 'svm_classification'):
        svm_classification(X, Y)
    elif(algo == 'random_forest_regressor'):
        random_forest_regressor(X, Y)
    elif(algo == 'svr'):
        svr(X, Y)
    else:
        print('Invalid DID')

def train_model():
    pass

def test_model():
    pass

def get_results():
    pass

if __name__ == '__main__':

    job = get_job_details()
    did = job['algo']['did']
    X, Y = preprocess_data(job)

    did_mapping = {'07A7287F45471dA8d7BddC647d49f03a54672E38': 'decision_tree', 
                   '2907E1f782f59C7B515c80B2DDB9DaC388F377F5': 'mlp_classifier', 
                   '2331e8116b0acB7AC164d8B4F332Aa104Eb9790F': 'naive_bayes',
                   '52D80495e56CFB241fD9e06aF7b5B96a80Ba509F': 'k_means',
                   '47838B1A397ed620F51D079Cd456d6564f940aC8': 'knn',
                   '3476E489Fb00058298fC8959Cb535fD3C29612c2': 'linear_regression',
                   '50374cfC875D9a66Be3f795Ff52Cb7714819eb7A': 'logistic_regression',
                   'e90F3344D508d017564b8EB4BB7e2E7C858365aa': 'MLP_Regressor',
                   '03115a5Dc5fC8Ff8DA0270E61F87EEB3ed2b3798': 'svm_classification',
                   '87F1A31A008D9cBE5c49B06dDb608df56967Cd51': 'random_forest_regressor',
                   '60b71c78E17de953A84f3A5A876645Cf15c1A92b': 'svr',
                   '0': 'elasticnet',
                   '1': 't_sne',
                   '2': 'non_lin_svm',
                   '3': 'PCA',
                   '4': 'CART'            
                   }

    algo = did_mapping[did]
    build_model(algo)
    train_model()
    test_model()
    get_results()