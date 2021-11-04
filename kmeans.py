"""
kmeans.py

A python implementation of the implementation and the experimentation of the K-Means clustering algorithm as a means of detecting
anamolies in Carleton's energy data.

Articles referenced:
https://www.datatechnotes.com/2020/05/anomaly-detection-with-kmeans-in-python.html
https://medium.datadriveninvestor.com/outlier-detection-with-k-means-clustering-in-python-ee3ac1826fb0

Written by Dominic Enriquez in collaboration with Ben Preiss
2 November 2021
Last Modified: 3 November 2021
"""

import os
import sys

#Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#K-Means Clustering Libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist

#PostgreSQL interactivity libraries
from psycopg2 import connect, sql
from datetime import datetime

def get_distances_and_points(data, centers, labels):
    '''
    '''
    distances_by_cluster = []
    indices_by_cluster = []
    for i, center in enumerate(centers):
        cluster = np.where(labels == i)[0]
        distance = np.sqrt((cluster - center) ** 2)
        distances_by_cluster.append(distance)
        indices_by_cluster.append(cluster)
    return distances_by_cluster, indices_by_cluster

def get_outliers(distances, indices, percentile):
    '''
    '''
    outliers = []
    for i in range(len(distances)):
        outlier_indices = indices[i][np.where(distances[i] > np.percentile(distances[i], percentile))]
        outliers.append(outlier_indices)
    return np.array(outliers).flatten()

def k_means(data, k = 3, plot = True, percentile = 90):
    '''
    '''
    x_axis = range(data.shape[1])
    room_temp = data[1,:]
    room_temp = room_temp.reshape(-1, 1)

    kmeans = KMeans(n_clusters = k).fit(room_temp)
    
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    distances, indices = get_distances_and_points(room_temp, centers, labels)
    print(distances)
    outliers = get_outliers(distances, indices, percentile)
    print(outliers)

    values = room_temp[outliers]
    
    if plot:
        plt.plot(x_axis, room_temp)
        plt.scatter(outliers, values, color = 'r')
        plt.show()
    return kmeans

def main():
    points_array = np.genfromtxt('evans_points.csv', delimiter=',')
    k_means(points_array, k = 3)

if __name__ == '__main__':
    main()
