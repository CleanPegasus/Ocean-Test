import numpy as np
import os
import time
import json
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
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
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.model_selection import train_test_split
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
    f.write('Predictions: ' + str(predictions))
    f.write('Accuracy: ' + str(accuracy))
    f.write('Confusion Matrix: \n' + str(cm))
    f.write('Classification Report: \n' + str(report))    
    f.close()

def naive_bayes(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    expected = y_test
    predicted = model.predict(X_test)
    accuracy = accuracy_score(expected, predicted)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(predicted))
    f.close()

def k_means(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
    kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    y_kmeans = kmeans.fit_predict(X)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(y_kmeans))
    f.close()

def knn(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, prediction))

    f = open("/data/outputs/result.txt", "w")
    f.write(str(prediction))
    f.close()

def linear_regression(X, Y):
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

    regressor = LinearRegression()
    regressor.fit(X_Train, Y_Train)

    Y_Pred = regressor.predict(X_Test)
    
    f = open("/data/outputs/result.txt", "w")
    f.write(str(Y_Pred))
    f.close()


def logistic_regression(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(C=1e5)
    logreg.fit(X_train, y_train)

    prediction = logreg.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, prediction))

    f = open("/data/outputs/result.txt", "w")
    f.write(str(prediction))
    f.close()

def MLP_Regressor(X,Y):

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    regr = MLPRegressor(random_state = 1, max_iter = 500)
    regr.fit(X_train, Y_train)
    prediction = regr.predict(X_test)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(prediction))
    f.close()

def svm_classification(X, Y):

    X = scale(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.10, random_state=101)
    svm_linear = svm.SVC(kernel='linear')
    svm_linear.fit(x_train, y_train)
    predictions = svm_linear.predict(x_test)
    confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)
    class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(class_wise))
    f.close()

def random_forest_regressor(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    regr = RandomForestRegressor(n_estimators=100, max_depth=2)
    regr.fit(X_train.values.reshape(-1,1), Y_train)
    Y_pred = regr.predict(X_test.values.reshape(-1,1))

    f = open("/data/outputs/result.txt", "w")
    f.write(str(Y_pred))
    f.close()

def svr(X, Y):

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(Y)

    regressor = SVR(kernel='linear')
    regressor.fit(X,y)

    y_pred = sc_y.inverse_transform((regressor.predict(sc_X.transform(np.array([[6.5]])))))

    f = open("/data/outputs/result.txt", "w")
    f.write(str(y_pred))
    f.close()


if __name__ == '__main__':

    job = get_job_details()
    did = job['algo']['did']
    X, Y = preprocess_data(job)

    did_mapping = {'07A7287F45471dA8d7BddC647d49f03a54672E38': 'decision_tree', 
                   '2907E1f782f59C7B515c80B2DDB9DaC388F377F5': 'mlp_classifier', 
                   '2': 'naive_bayes',
                   '3': 'k_means',
                   '4': 'knn',
                   '5': 'linear_regression',
                   '6': 'logistic_regression',
                   '7': 'MLP_Regressor',
                   '8': 'svm_classification',
                   '9': 'random_forest_regressor',
                   '10': 'svr'
                   }

    algo = did_mapping[did]

    if(algo == 'decision_tree'):
        decision_tree(X, Y)
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


