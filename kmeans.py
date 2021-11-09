"""
kmeans.py

A python implementation of the implementation and the experimentation of the K-Means clustering algorithm as a means of detecting
anamolies in Carleton's energy data.

Articles referenced:
https://www.datatechnotes.com/2020/05/anomaly-detection-with-kmeans-in-python.html
https://medium.datadriveninvestor.com/outlier-detection-with-k-means-clustering-in-python-ee3ac1826fb0

Written by Dominic Enriquez in collaboration with Ben Preiss
2 November 2021
Last Modified: 4 November 2021
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

def generate_test_data():
    '''
    Generates a list of test data that follows a normal distribution centered at 65 
    with a standard deviation 15 to resemble temperature data.
    Each element in data is a list of 96 data points meant to correspond with temperature
    readings over time. There are 100 such rooms in the list
    '''
    data = []
    # for each hypothetical room that we want to track
    for _ in range(15):
        room = []
        # generate 96 temperature readings associated with a particular time index
        for i in range(10):
            a = np.random.normal(65, 15)
            room.append(a)
        data.append(room)
    return np.array(data)

def k_means(data, k = 1, plot = True):
    '''
    '''
    km = KMeans(n_clusters = k)
    distance_from_all_clusters = km.fit_transform(data)
    min_distances = [min(distances) for distances in distance_from_all_clusters]
    sum_distances = [sum(distances) for distances in distance_from_all_clusters]

    mean = np.mean(min_distances)
    sd = np.std(min_distances)

    indices = []
    for room_index, distance in enumerate(min_distances):
        if (distance > (mean + 2 * sd)) or (distance < (mean - 2 * sd)):
            indices.append(room_index)
    print(len(indices))
    for index, temperature_data in enumerate(data):
        if index in indices:
            #make it red
            times = []
            temps = []
            for time, temp in enumerate(temperature_data):
                times.append(time)
                temps.append(temp)
            plt.plot(times, temps, c = 'r')
        else:
            times = []
            temps = []
            for time, temp in enumerate(temperature_data):
                times.append(time)
                temps.append(temp)
            plt.plot(times, temps, c = 'k')
    plt.show()
    if plot:
        pass

def main():
    data = generate_test_data()
    k_means(data)


if __name__ == '__main__':
    main()
