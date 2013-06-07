#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import numpy as np
import scipy

#clustering
#cf. http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics

def create_clusters(documents, cluster_algorithm, cluster_options):
    """
    Clusters the /documents/ using /cluster_algorithm/ with /cluster_options/.
    """
    algorithms = {'kmeans':KMeans, 'dbscan':DBSCAN}
    if cluster_algorithm not in algorithms.keys():
        logging.error('Cluster algorithm "{alg}" unknown! Won\'t cluster.'.format(
            cluster_algorithm))

    doc_matrix, doc_list = documents.as_centroid_matrix()
    
    cluster_alg = algorithms[cluster_algorithm]
    cluster_res = perform_clustering(cluster_alg, doc_matrix, cluster_options)
    clusters = process_clustering(cluster_res, doc_list, documents)
    return clusters


def create_clusters_dbscan_infer_options(documents, X=1):
    """
    Returns dbscan clustering results using inferred eps and minsample options.
    eps is the average of the documents' distances and minsample is X% of the
    number of data points.
    """
    doc_matrix, doc_list = documents.as_centroid_matrix()
    matrix = compute_distance_matrix(doc_matrix)
    
    #estimate min_samples
    X = 0.01 * X #percentage
    min_samples = X * len(doc_list)
    #estimate eps
    avg = np.average(matrix)
    #median = np.median(matrix)
    #eps is "the maximum distance between two samples for them to be considered as
    #in the same neighborhood." setting the upper bound based on matrix
    cluster_options = {'eps':round(avg*4 ,2), 'min_samples':min_samples}
    #cluster_options = {'eps':round(median*5 ,2), 'min_samples':min_samples}

    cluster_alg = DBSCAN
    logging.info('Cluster algorithm is {alg}; options estimation is {options}'.format(
        alg=cluster_alg.__name__, options=cluster_options))

    cluster_res = cluster_alg(**cluster_options).fit(matrix)
    clusters = process_clustering(cluster_res, doc_list, documents)
    return clusters


def perform_clustering(cluster_alg, matrix, cluster_options):
    """Used /alg/ to cluster the data in /similarity_matrix/."""
    logging.info('Cluster algorithm is {alg}; options are {options}'.format(
        alg=cluster_alg.__name__, options=cluster_options))
    matrix = compute_distance_matrix(matrix)
    cluster_res = cluster_alg(**cluster_options).fit(matrix)
    return cluster_res


def compute_distance_matrix(matrix):
    """
    Computes distance matrix D for the documents and returns it together with
    doc_list that knows which line in D corresponds to which document.
    """
    D = distance.squareform(distance.pdist(matrix, 'euclidean'))
    return D


def compute_similarity_matrix(matrix):
    """
    Computes similarity matrix S for the documents and returns it together with
    doc_list that knows which line in S corresponds to which document.
    """
    D = compute_distance_matrix(documents)
    S = 1 - (D / np.max(D))
    return S


def process_clustering(cluster_res, doc_list, documents):
    """Takes the cluster result and creates a more readable representation."""
    labels = cluster_res.labels_
    #Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    clusters = {}
    for k in set(labels):
        class_members = [index[0] for index in np.argwhere(labels == k)]
        member_docs = []
        count = 0
        for m in class_members:
            doc = doc_list[m]
            #store the cluster information in the doc object
            documents.docs[doc].set_cluster_assignment(k+1) #cluster names start at 1
            member_docs.append(doc)
            count += 1
        if k != -1:
            clusters['cluster{k} #{count}'.format(k=int(k)+1,count=count)] = member_docs
        else:
            logging.info('Could not cluster {count} documents'.format(count=count))
    return clusters
