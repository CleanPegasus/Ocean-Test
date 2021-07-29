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
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.svm import SVR
from sklearn import decomposition
from sklearn import linear_model
from sklearn import tree
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
    return (X, Y)

def decode_results(encode, Y):
    global y_le
    if encode:
        Y = list(y_le.inverse_transform(Y))

    return(Y)

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

    y_pred = decode_results(encode, y_pred)

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
    predictions = decode_results(encode, predictions)
    
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
    predicted = decode_results(encode, predicted)

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
    y_kmeans = decode_results(encode, y_kmeans)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(y_kmeans))
    f.close()

def knn(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, prediction))
    prediction = decode_results(encode, prediction)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(prediction))
    f.close()

def linear_regression(X, Y):
    
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

    regressor = LinearRegression()
    regressor.fit(X_Train, Y_Train)

    Y_Pred = regressor.predict(X_Test)
    Y_Pred = decode_results(encode, Y_Pred)
    
    f = open("/data/outputs/result.txt", "w")
    f.write(str(Y_Pred))
    f.close()


def logistic_regression(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(C=1e5)
    logreg.fit(X_train, y_train)

    prediction = logreg.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, prediction))
    prediction = decode_results(encode, prediction)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(prediction))
    f.close()

def MLP_Regressor(X,Y):

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    regr = MLPRegressor(random_state = 1, max_iter = 500)
    regr.fit(X_train, Y_train)
    prediction = regr.predict(X_test)
    prediction = decode_results(encode, prediction)

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
    predictions = decode_results(encode, predictions)
    class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(class_wise))
    f.close()

def random_forest_regressor(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    regr = RandomForestRegressor(n_estimators=100, max_depth=2)
    regr.fit(X_train.values.reshape(-1,1), Y_train)
    Y_pred = regr.predict(X_test.values.reshape(-1,1))
    Y_pred = decode_results(encode, Y_pred)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(Y_pred))
    f.close()

def svr(X, Y):

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    Y = sc_y.fit_transform(Y)

    regressor = SVR(kernel='linear')
    regressor.fit(X,Y)

    y_pred = sc_y.inverse_transform((regressor.predict(sc_X.transform(X_test))))
    y_pred = decode_results(encode, y_pred)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(y_pred))
    f.close()

def elasticnet(X, Y):
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.15)

    alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

    for a in alphas:
        model = ElasticNet(alpha=a).fit(X,Y)   
        score = model.score(X, Y)
        pred_y = model.predict(X)
        mse = mean_squared_error(Y, pred_y)   
        print("Alpha:{0:.4f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
        .format(a, score, mse, np.sqrt(mse)))

    elastic=ElasticNet(alpha=0.01).fit(xtrain, ytrain)
    ypred = elastic.predict(xtest)
    score = elastic.score(xtest, ytest)
    mse = mean_squared_error(ytest, ypred)
    print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
        .format(score, mse, np.sqrt(mse)))

    ypred = decode_results(encode, ypred)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(ypred))
    f.close()

    # x_ax = range(len(xtest))
    # plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
    # plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
    # plt.legend()
    # plt.savefig('/data/outputs/plot.png')

# def plot_iris_2d(x, y, title):    
#         plt.scatter(x, y,c=y,s=70)
#         plt.title(title, fontsize=20, y=1.03)
#         # plt.show()
#         plt.savefig('/data/outputs/plot.png')

def t_SNE(X,Y):    

    tsne = TSNE(n_components=2, n_iter=1000, random_state=42)
    points = tsne.fit_transform(X)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(points))
    f.close()

    # plot_iris_2d(
    #     x = points[:, 0],
    #     y = points[:, 1],
    #     title = 'Iris dataset visualized with t-SNE')



def non_lin_svm(X,Y):
    
    X = scale(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.10, random_state=101)

    svm_non_linear = svm.NuSVC(gamma = 'auto')
    svm_non_linear.fit(x_train, y_train)

    predictions = svm_non_linear.predict(x_test)
    confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)
    class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
    predictions = decode_results(encode, predictions)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(class_wise) + '\n' + str(predictions))
    f.close()

def PCA(X,Y):

    pca = decomposition.PCA(n_components=X.shape[0])
    pca.fit(X)
    X = pca.transform(X)

    f = open("/data/outputs/result.txt", "w")
    f.write(str(X))
    f.close()


def CART(X,Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

    clf= DecisionTreeClassifier(random_state = 100)
    clf.fit(X_train, y_train)

    Y_pred=clf.predict(X_test)
    print(Y_pred)
    acc = accuracy_score(y_test, Y_pred)
    print("Accuracy:",acc)
    
    cm=np.array(confusion_matrix(y_test,Y_pred))
    print(cm)
    
    Y_pred = decode_results(encode, Y_pred)

    text_representation = tree.export_text(clf)
    print(text_representation)
    f = open("/data/outputs/result.txt", "w")
    f.write('Text Representation: \n'+str(text_representation))
    f.write('Accuracy: '+str(acc))
    f.write('Confusion Matrix: \n'+str(cm))
    f.write('Prediction: '+str(Y_pred))
    f.close()


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
    else:
        print('Invalid DID')