# Uses IRIS dataset
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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

def plot_iris_2d(x, y, title):    
        plt.scatter(x, y,c=y,s=70)
        plt.title(title, fontsize=20, y=1.03)
        # plt.show()
        plt.savefig('/data/outputs/plot.png')

def t_SNE(job_details):
    print('Starting compute job with the following input information:')
    print(json.dumps(job_details, sort_keys=True, indent=4))

    first_did = job_details['dids'][0]
    filename = job_details['files'][first_did][0]

    dataset = pd.read_csv(filename)
    X = dataset.drop(["target"], axis = 1)
    y = dataset['target']

    tsne = TSNE(n_components=2, n_iter=1000, random_state=42)
    points = tsne.fit_transform(X)

    plot_iris_2d(
        x = points[:, 0],
        y = points[:, 1],
        title = 'Iris dataset visualized with t-SNE')
    

if __name__ == '__main__':
    t_SNE(get_job_details())
