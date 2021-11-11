"""
kmeans.py

A python implementation of the implementation and the experimentation of the K-Means clustering algorithm as a means of detecting
anamolies in Carleton's energy data.

Articles referenced:
https://www.datatechnotes.com/2020/05/anomaly-detection-with-kmeans-in-python.html
https://medium.datadriveninvestor.com/outlier-detection-with-k-means-clustering-in-python-ee3ac1826fb0

Written by Dominic Enriquez and Ben Preiss
2 November 2021
Last Modified: 11 November 2021
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

def generate_test_data():
    '''
    Generates a list of test data that follows a normal distribution centered at 65 
    with a standard deviation 15 to resemble temperature data.
    Each element in data is a list of 96 data points meant to correspond with temperature
    readings over time. There are 100 such rooms in the list
    '''
    data = []
    # for each hypothetical room that we want to track
    for _ in range(100):
        room = []
        # generate 15 temperature readings associated with a particular time index
        for _ in range(15):
            room.append(np.random.normal(65, 5))
        data.append(room)
    return np.array(data)

def k_means(data, k = 3):
    '''
    '''
    km = KMeans(n_clusters = k)
    distance_from_all_clusters = km.fit_transform(data)
    min_distances = [min(distances) for distances in distance_from_all_clusters]
    #sum_distances = [sum(distances) for distances in distance_from_all_clusters]

    mean = np.mean(min_distances)
    sd = np.std(min_distances)

    anomalous_rooms = []
    for room_index, distance in enumerate(min_distances):
        if (distance > (mean + 2.3 * sd)):
            print("Room %d is anomalous" % room_index)
            anomalous_rooms.append(room_index)
    for room_index, temperature_data in enumerate(data):
        times = []
        temps = []
        for time, temp in enumerate(temperature_data):
            times.append(time)
            temps.append(temp)
        if room_index in anomalous_rooms:
            #make anamolies red
            plt.plot(times, temps, c = 'r', zorder = 2)
        else:
            plt.plot(times, temps, c = 'k', zorder = 1)
    plt.ylim(30,100)
    if (len(anomalous_rooms) == 0):
        print("No anomalies detected")
    plt.show()

def main():
    data = generate_test_data()
    k_means(data)


if __name__ == '__main__':
    main()

