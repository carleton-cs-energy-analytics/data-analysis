"""
kmeans.py

A python implementation of the implementation and the experimentation of the K-Means clustering algorithm as a means of detecting
anamolies in Carleton's energy data.

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

#PostgreSQL interactivity libraries
from psycopg2 import connect, sql
from datetime import datetime

def get_distance_from_nearest_cluster(data, centers):
    distances = []
    for data_point in data:
        for center in centers:
            distances_from_each_center = []
            distances_from_each_center.append(np.sqrt((data_point[0] - center[0]) ** 2))
        min_distance = min(distances_from_each_center)
        distance_from_cluster = np.array(min_distance)
        distances.append(distance_from_cluster)
    return np.array(distances)

def k_means(data, n = 5, k = 3, plot = True):
    x_axis = range(data.shape[1])
    room_temp = data[1,:]
    room_temp = room_temp.reshape(-1, 1)

    kmeans = KMeans(n_clusters = k).fit(room_temp)
    
    centers = kmeans.cluster_centers_

    distances = get_distance_from_nearest_cluster(room_temp, centers)
    print(distances)
    ordered_indecies = np.argsort(distances, axis = 0)
    indecies = ordered_indecies[-n:]
    values = room_temp[indecies]

    if plot:
        plt.plot(x_axis, room_temp)
        plt.scatter(indecies, values, color = 'r')
        plt.show()
    return kmeans

def main():
    points_array = np.genfromtxt('evans_points.csv', delimiter=',')
    kmeans = k_means(points_array, k = 1)

if __name__ == '__main__':
    main()
