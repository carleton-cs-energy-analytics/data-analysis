"""
1means.py

A python implementation of the implementation and the experimentation of the K-Means clustering algorithm as a means of detecting
anamolies in Carleton's energy data. Currently only works for k = 1.

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

def k_means(data, n = 5, plot = True):
    x_axis = range(data.shape[1])
    room_temp = data[1,:]
    room_temp = room_temp.reshape(-1, 1)

    kmeans = KMeans(n_clusters = 1).fit(room_temp)

    centers = kmeans.cluster_centers_

    distance = np.sqrt((room_temp - centers)**2)
    ordered_indecies = np.argsort(distance, axis = 0)
    indecies = ordered_indecies[-n:]
    values = room_temp[indecies]

    if plot:
        plt.plot(x_axis, room_temp)
        plt.scatter(indecies, values, color = 'r')
        plt.show()
    return kmeans

def main():
    points_array = np.genfromtxt('evans_points.csv', delimiter=',')
    kmeans = k_means(points_array)

if __name__ == '__main__':
    main()
