# stl.py
#Created by Owen Szafran 01/2021
#Last Modified: 4/29/2021

#libraries for moving files to frontend
import os
import sys

#importing plotting libraries
import pylab
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#matplotlib.use('AGG')

#importing libraries needed for stldecompose
import statsmodels.api as sm
import pandas as pd
from stldecompose import decompose, forecast 
from stldecompose.forecast_funcs import (drift, seasonal_naive)
from sklearn.ensemble import IsolationForest

# import the psycopg2 database adapter for PostgreSQL
import psycopg2
from psycopg2 import connect, sql
from datetime import datetime

# import the sys library for arguments
import sys

#creating a connection to the database server
conn = psycopg2.connect(
    host="localhost",
    database="energy",
    user="energy",
    password="less!29carbon")
EVANS_POINTS_FILE = "/var/www/backend/data-analysis/evans_points.csv"
CSV_DIRECTORY = "/var/www/frontend/static/csv-files"
IMAGES_DIRECTORY = "/var/www/frontend/static/images"
def create_csv(building_name):
    df = pd.DataFrame(dtype=str)
    cur = conn.cursor()
    '''
    Description: Gets all point ID's of three specific points (Room Temp, Valve, and Virtual Room Temp Setpoint) from the Database
    for each room that has those points in a certain building. Creates a csv containing the rooms and their points.
    '''
    building_query = '''SELECT building_id FROM buildings WHERE name = '{0}' '''.format(building_name)
    cur.execute(building_query)
    building_id = cur.fetchone()[0]

    # input the tag_ids of the three types of points you want from each room
    tag_id1 = 3 # Room Temperature
    tag_id2 = 6 # Valve
    tag_id3 = 10 # Virtual Room Temperature Setpoint

    # Gets all the rooms and point_ids for rooms with 
    all_query = '''WITH DR as (SELECT D.device_id, R.name from rooms as R JOIN devices as D ON D.room_id = R.room_id WHERE R.building_id = {0} AND R.name NOT LIKE 'UnID%')
                    SELECT P3.room, pid1, pid2, pid3 FROM (SELECT P1.point_id as pid1, P2.point_id as pid2, P2.name as room FROM (SELECT P.point_id, DR.name FROM points as P
                    JOIN DR ON DR.device_id = P.device_id WHERE P.point_id IN (SELECT point_id FROM points_tags WHERE tag_id = {1})) as P1
                    JOIN (SELECT P.point_id, DR.name FROM points as P JOIN DR ON DR.device_id = P.device_id WHERE P.point_id IN (SELECT point_id FROM points_tags WHERE tag_id = {2})) as P2 ON P1.name=p2.name) as P4
                    JOIN (SELECT P.point_id as pid3, DR.name as room FROM points as P JOIN DR ON DR.device_id = P.device_id WHERE P.point_id IN (SELECT point_id FROM points_tags WHERE tag_id = {3})) as P3 ON P3.room=P4.room ORDER BY P3.room'''.format(building_id, tag_id1, tag_id2, tag_id3)
    cur.execute(all_query)
    everything = cur.fetchall()
    conn.close()

    for room in everything:
        name = room[0]
        lst = [room[1], room[2], room[3]]
        df[name] = lst
    
    file_name = building_name.lower().replace(" ", "_") + "_points.csv"
    df.to_csv(file_name, index=False)


	
def values_in_last_n_days(point_id, days, days_ago=0):
    '''
    Description: creating and running a query for a point from the API
    Parameters: int point_id to be queried, int number of days to get values from
    Returns: a list of the point's values
    '''
    cur = conn.cursor()
    limit = 96*days
    offset_value = 96*days_ago
    query = '''SELECT *
               FROM values
               WHERE point_id = %s
               ORDER BY timestamp DESC
               LIMIT %s OFFSET %s'''
    cur.execute(query, (point_id, limit, offset_value))
    values = cur.fetchall()
    cur.close()
    return values

def create_float_series(point_id, days):
    '''
    Description: creates a dataframe of values for a point with values stored as floats in the database by calling values_in_last_n_days. Splits the data frame and corresponding indexes into a section for the last day and a section for data before that.
    Parameters: int point_id to be queried, int number of days to get values from
    Returns: Dataframes for fullseries, detection series (the last day), and prior series (values before the last day), array indices corresponding to each dataframe
    '''
    values = values_in_last_n_days(point_id, days, days_ago=337)#to get historical data, use the optional days_ago variable here
    times_arr = [] 
    vals_arr = []
    for value in values:
        time = value[2]
        val = value[4]
        times_arr.append(datetime.utcfromtimestamp(time).strftime("%Y-%m-%d %H:%M:%S"))
        vals_arr.append(val)
    forward_times_arr = times_arr[::-1]
    forward_vals_arr = vals_arr[::-1]
    d = {'value': forward_vals_arr}
    series_df = pd.DataFrame(data=d, index=forward_times_arr, dtype=float)
    prior_df = series_df.iloc[:-96:]
    detection_df = series_df.iloc[-96::]
    prior_index = forward_times_arr[:-96:]
    detection_index = forward_times_arr[-96::]
    return (series_df, prior_df, detection_df, forward_times_arr, prior_index, detection_index)

def series_decompose(series_df):
    '''
    Description: Runs the stldecompose decompose method on a time series from create_float_series by adding 96 (number of points in a day in the database) as period value
    Parameters: A pandas DataFrame time series
    Returns: a statsmodel object representing the decomposed series 
    '''
    decomped_series = decompose(series_df.values, period=96)
    return decomped_series

def get_components(decomp):
    '''
    Parameters: a statsmodel object representing a decomposed series
    Returns: numpy arrays of the various trend components
    '''
    t = decomp.trend
    s = decomp.seasonal
    r = decomp.resid
    return (t, s, r)

def stl_float(point_id, days, plot_ser=False, plot_comp=False):
    '''
    Description: runs the stl algorithm for a point and a day by calling create_float_series and decomposing the series. Plots the original series and the component series when optional parameters are set.
    Parameters: a point_id, days, and optional boolean plot values
    Returns: the dataframe time series of the point, statsmodel object of the decomposed series, dataframes for the detection and prior periods, and corresponding index arrays
    '''
    series_df, prior_df, detection_df, index, prior_index, detection_index = create_float_series(point_id, days)
    decomp = series_decompose(series_df)
    if plot_ser:
        plot_series(series_df, index, point_id, days)
    if plot_comp:
        plot_components(decomp, index, point_id, days)
    return series_df, decomp, prior_df, detection_df, index, prior_index, detection_index

def detect_evans_anomalies(days=5, anom_thres=36,  plot_ser=False, plot_comp=False, plot_anom=False):
    '''
    Description: Method to run stl algorithm on all rooms in evans and generate corresponding graphs by calling detect_room_anomalies. Gets the point ids from the evans_points.csv file.
    Parameters: all optional, an int number of days, an int limit on anomaly points, and boolean plot values
    Returns: dataframes with rooms as column indeces and point types as row indeces, one containing anomaly counts and the other containing a list of the anomalies for each room/point
    '''
    building_df = pd.read_csv(EVANS_POINTS_FILE, dtype=str)
    building_df.index=['temp', 'vent', 'virtual set']
    anom_counts = pd.DataFrame(index=['Room Temperature', 'Valve Angle',  'System Set Temperature and Room Temperature Difference'], dtype=int) 
    anom_rooms = pd.DataFrame(index=['temp', 'vent', 'set temp diff'], dtype=str)    
    
    for column in building_df:
        room_ids = building_df.loc[:,column]
        room_anoms, num_room_anoms = detect_evans_room(room_ids, days, building_name="EV", plot_ser=plot_ser, plot_comp=plot_comp, plot_anom=plot_anom)
        anom_counts[column] = num_room_anoms
        anom_rooms[column] = [anomalies_report_string(room_anoms[0]), anomalies_report_string(room_anoms[1]), anomalies_report_string(room_anoms[2])]

    forest_arr = isolation_forest(building_df, days)
    forest_arr.columns = anom_counts.columns
    anom_counts = anom_counts.append(forest_arr)
    return anom_rooms, anom_counts

def isolation_forest(building_df, days):
    '''
    Description: A method to run isolation forest anomaly detection on a given building. 
    Parameters: The dataframe of points for a building created by create_building_csv, the number of days. 
    '''
    temp_arr = []
    for column in building_df:
        room_ids = building_df.loc[:,column]
        temp_id = room_ids.loc['temp']
        vsp_id = room_ids.loc['virtual set']
        temp_series, temp_prior_df, temp_detection_df, index, prior_index, detection_index = create_float_series(temp_id, days)
        vsp_series, vsp_prior_df, vsp_detection_df, vsp_index, vsp_prior_index, vsp_detection_index = create_float_series(vsp_id, days)
        diff_series = vsp_series.sub(temp_series).fillna(value=0, axis=0)[-1*days*96::4]
        diff_arr = []
        for i in range(len(diff_series)):
            diff_arr.append(diff_series.values[i][0])
        temp_arr.append(diff_arr)
    clf = IsolationForest(random_state=0).fit(temp_arr)
    unweighted = clf.predict(temp_arr)
    weighted = pd.DataFrame(index=['Forest Anomaly Score'], dtype=str)
    for i in range(len(unweighted)):
        if unweighted[i] == 1:
            weighted[i] = 0
        else:
            weighted[i] = 10
  
    return weighted

def detect_evans_room(ids, days, building_name="", plot_ser=False, plot_comp=False, plot_anom=False):
    '''
    Description: runs stl on the given points in the room. MODIFY THIS METHOD to change which series stl is run on. Currently running on: room temperature, vent angle, difference between set and room temp, and difference between set and virtual set values. 
    Parameters: an array of int point id's for a given room and a number of days. Optional boolean plot settings. 
    Returns: A list of lists of anomalies for each point associated with the room, and a list of counts of anomalies by point
    '''
    room = ids.name
    temp_id = ids.loc['temp']
    vent_id = ids.loc['vent']
    vsp_id = ids.loc['virtual set']
    temp_name_str = building_name + str(room) + 'Temp'
    temp_anom = temp_anomalies(temp_name_str, temp_id, days, plot_anom=plot_anom)
    vent_name_str = building_name + str(room) + 'ValveAngle'
    vent_anom = valve_anomalies(vent_name_str, vent_id, days, plot_anom=plot_anom) 
    diff_name_str = building_name + str(room) + 'VSetTempDiff'
    diff_anom = virtset_temp_diff_anomalies(diff_name_str, vsp_id, days, second_point_id=temp_id, plot_anom=plot_anom)
    num_temp_anom = len(temp_anom)
    num_diff_anom = len(diff_anom)
    num_vent_anom = len(vent_anom)
    num_room_anoms = [num_temp_anom, num_vent_anom, num_diff_anom]
    room_anoms = [temp_anom, vent_anom, diff_anom]
    return room_anoms, num_room_anoms

def temp_anomalies(name_str, point_id, days, plot_anom=False, heur_upperbound=80, heur_lowerbound=60, heur_emergency_temp=40):
    '''
    Description: A mathod to perform STL and Heuristic anomaly detection on the temperature values of a given room. 
    Parameters: the room name string, point id, number of days, plot anomalies boolean values, heuristic upper, lower, and emergency bounds
    TODO: set up email notifications with rthe emergency temperature!
    '''
    series_df, decomp, prior_df, detection_df, index, prior_index, detection_index = stl_float(point_id, days)
    stl_expected, greaterbound, lesserbound = get_expected_bounds(decomp, prior_df)
    greaterbound = add_heur_to_bound(greaterbound, heur_upperbound)
    lesserbound = add_heur_to_bound(lesserbound, heur_lowerbound, upper=False)
    anomalies = find_anomalies(greaterbound, lesserbound, detection_df, detection_index)
    if plot_anom:
        plot_anomalies(name_str, series_df, index, point_id, days, predicted=stl_expected, upperbound=greaterbound, lowerbound=lesserbound, heur_upper = heur_upperbound, heur_lower=heur_lowerbound, emerg_bound=heur_emergency_temp) 
    return anomalies

def valve_anomalies(name_str, point_id, days, plot_anom=False, heur_limit=95, heur_limit_time=1):
    '''
    Description:  A method to perform heuristic anomaly detetion on the valve angle values of a given room.
    Parameters: the room name str, point id, number of days, plot anomalies boolean value, heuristic upper bound, and the heuristic time limit in hours. 
    '''
    series_df, prior_df, detection_df, index, prior_index, detection_index = create_float_series(point_id, days)
    beyond_lim_count = 0
    anomalies = []
    for i in range (96):
        if (detection_df.values[i] >= heur_limit):
            beyond_lim_count += 1
        else: 
            beyond_lim_count = 0
        if (beyond_lim_count >=4):
            anomalies.append((detection_index[i], 'Valve too open for too long', detection_df.values[i]))
    if plot_anom:
        plot_anomalies(name_str, series_df, index, point_id, days, heur_upper=heur_limit)
    return anomalies

def virtset_temp_diff_anomalies(name_str, point_id, days, second_point_id, plot_anom=False, heur_upper_limit=5, heur_lower_limit=-5, heur_time_limit=1):
    '''
    Description: A method to perform STL and heuristic anomaly detetion on the virtual set temp and actual room temp difference of a room. 
    Parameters: the room name string, point Id's, number of days, plot anomalies boolean value, heuristic upper and lower bound, and the heuristic time threshold.  
    '''
    series_df, decomp, prior_df, detection_df, index, prior_index, detection_index, point_id = stl_diff(point_id, second_point_id, days)
    stl_expected, greaterbound, lesserbound= get_expected_bounds(decomp, prior_df)
    greaterbound = add_heur_to_bound(greaterbound, heur_upper_limit)
    lesserbound = add_heur_to_bound(lesserbound, heur_lower_limit, upper=False)
    anomalies = find_anomalies(greaterbound, lesserbound, detection_df, detection_index)
    if plot_anom:
        plot_anomalies(name_str, series_df, index, point_id, days, predicted=stl_expected, upperbound=greaterbound, lowerbound=lesserbound, heur_upper = heur_upper_limit, heur_lower=heur_lower_limit)
    return anomalies


def add_heur_to_bound(bound, heur, upper=True):
    '''
    Description: A method to create bounds combining the STL bound and the heuristic bound. 
    Parameters: stl bound dataframe, the heuristic int value, and a boolean upper value, that sets whether this is creating an upper or lower bound
    '''
    for i in range(96):
        if (upper):
            if (bound[i] > heur):
                bound[i] = heur    
        else:
            if (bound[i] < heur):
                bound[i] = heur 
    return bound



def stl_diff(minu_point_id, subt_point_id, days, plot_ser=False, plot_comp=False):
    '''
    Description: runs the stl algorithm on a time series of a difference between two points, by calling create_float_series and decomposing the series. Plots the original series and the component series when optional parameters are set.
    Parameters: a point_id, a second point_id, days, and optional boolean plot values
    Returns: the dataframe time series of the point, statsmodel object of the decomposed series, dataframes for the detection and prior periods, and corresponding index arrays
    '''
    minu_series, minu_prior_df, minu_detection_df, index, prior_index, detection_index = create_float_series(minu_point_id, days)
    subt_series, subt_prior_df, subt_detection_df, subt_index, subt_prior_index, subt_detection_index = create_float_series(subt_point_id, days)
    diff_series = minu_series.sub(subt_series).fillna(value=0, axis=0)[-1*days*96::]
    diff_prior_series = minu_prior_df.sub(subt_prior_df).fillna(value=0, axis=0)[-1*days*96::]
    diff_detection_series = minu_detection_df.sub(subt_detection_df).fillna(value=0, axis=0)[-1*days*96::]
    decomp = series_decompose(diff_series)
    diff_point_id = str(minu_point_id) + "-" + str(subt_point_id)
    if plot_ser:
        plot_series(diff_series, index, diff_point_id, days)
    if plot_comp:
        plot_components(decomp, index, diff_point_id, days)
    return diff_series, decomp, diff_prior_series, diff_detection_series, index, prior_index, detection_index, diff_point_id


def stl_point_anomalies(name_id, point_id, days, second_point_id=None, plot_ser=False, plot_comp=False, plot_anom=False):
    '''
    Description: Runs stl on either a point or two points by calling either stl_float or stl_diff based on the number of parameters provided. Optionally plots the anomaly graph. 
    Parameters: a point_id, days, optional second point id and boolean plot settings
    Returns: a list of all anomalies found.
    '''
    if second_point_id!=None:
        series_df, decomp, prior_df, detection_df, index, prior_index, detection_index, point_id = stl_diff(point_id, second_point_id, days, plot_ser=plot_ser, plot_comp=plot_comp)
    else:
        series_df, decomp, prior_df, detection_df, index, prior_index, detection_index = stl_float(point_id, days, plot_ser=plot_ser, plot_comp=plot_comp)
    stl_expected, greaterbound, lesserbound= get_expected_bounds(decomp, prior_df)
    anomalies = find_anomalies(greaterbound, lesserbound, detection_df, detection_index)
    if plot_anom:
        plot_anomalies(name_id, series_df, stl_expected, greaterbound, lesserbound, index, point_id, days)
    return anomalies

def anomalies_report_string(anomalies):
    '''
    Parameters: a list of anomalies
    Returns: a formatted string reporting the anomalies
    '''
    report = ""
    for anomaly in anomalies:
        report = report + "Anomaly at " + str(anomaly[0]) + ": value too " + anomaly[1] + ". (bound, anomaly_value): (" + str(anomaly[2]) +")\n"
    return report

def find_anomalies(greaterbound, lesserbound, series_df, index):
    '''
    Description: Iterates over the detection series (values from the last day) and adding anomalies where it is outside of the established bounds
    Parameters: series representing the upper and lower bounds, the series dataframe for the last day, and corresponding indeces
    Returns: A list of anomalies
    '''
    anomalies = []
    for i in range(96):
        if (min(lesserbound[i], greaterbound[i]) > series_df.values[i]):
            anomalies.append((index[i], 'low', min(lesserbound[i], greaterbound[i]), series_df.values[i]))
        elif (max(lesserbound[i], greaterbound[i]) < series_df.values[i]):
            anomalies.append((index[i], 'high', max(lesserbound[i], greaterbound[i]), series_df.values[i]))
    return anomalies

def get_expected_bounds(decomp, prior_df):
    '''
    Description: Generates upper and lower bounds for the detection series, and the expected series for the entire time frame
    Parameters: statsmodel object of decomposed series, the prior_df
    Returns: expected series and greater and lesserbounds
    '''
    trend, seasonal, resid = get_components(decomp)
    prior_resid = get_components(series_decompose(prior_df))[2]
    stl_expected = trend + seasonal
    std_dev = np.std(prior_resid)
    delta = 2*std_dev
    detection_expected = stl_expected[-96::]
    greaterbound = detection_expected + delta
    lesserbound = detection_expected - delta
    return stl_expected, greaterbound, lesserbound

def plot_series(series_df, index, point_id, days):
    '''
    Description: Method to plot the original series of a point
    Parameters: series dataframe, datetime index, point_id and days
    '''
    s = series_df.values.tolist()
    plt.figure(figsize=(70,45))
    plt.rcParams.update({'font.size': 40})
    plt.plot(index, s, marker='', linestyle='-', label='Values', linewidth=4)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.xticks(np.arange(0,len(index),step = len(index)/(days*4)), index[0::24], rotation=45)
    plt.grid(axis='both')
    plt.legend()
    plt.title("Observed Values of Point {0} over last {1} days".format(point_id, days))
    plt.savefig("point{0}series{1}".format(point_id, index[len(index)-1][:10]))


def plot_anomalies(name_id, series_df, index, point_id, days, predicted=None, upperbound=None, lowerbound=None, heur_upper=None, heur_lower=None, emerg_bound=None):
    '''
    Description: Saves a png file of the plot of the series, expected value for the entire series, and the bounds for the detection time frame
    Parameters: the series dataframe, the expected series and bounds arrays, datetime index array,  point_id and days
    '''
    s = series_df.values.tolist() 
    plt.figure(figsize=(70,45))
    plt.rcParams.update({'font.size': 40})
    plt.plot(index, s, marker='', linestyle='-', label='Values', linewidth=4)
    if (predicted is not None):
        #e = predicted.tolist()
        e = predicted
        plt.plot(index, e, marker='', linestyle='-', label='Predicted', linewidth=4)
    if (upperbound is not None):
        u = upperbound.tolist()
        plt.plot(index[-96::], u, marker='', linestyle='-', label='Upper Bound', linewidth=4)
    if (lowerbound is not None):
        l = lowerbound.tolist()
        plt.plot(index[-96::], l, marker='', linestyle='-', label='Lower Bound', linewidth=4)
    if (heur_upper is not None):
        h_u = [heur_upper] * 96
        plt.plot(index[-96::], h_u, marker='', linestyle='-', label='Max Bound', linewidth=4)
    if (heur_lower is not None):
        h_l = [heur_lower] * 96
        plt.plot(index[-96::], h_l, marker='', linestyle='-', label='Min Bound', linewidth=4)
    if (emerg_bound is not None):
        e_b = [emerg_bound] * 96
        plt.plot(index[-96::], e_b, marker='', linestyle='-', label='Emergency', linewidth=4)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.axvline(x=index[-96])
    plt.xticks(np.arange(0,len(index),step = len(index)/(days*4)), index[0::24], rotation=45)
    plt.grid(axis='both')
    plt.legend()
    plt.title("{0} over last {1} days".format(name_id, days))
    filename = "{0}Anomalies".format(name_id)
    plt.savefig(filename)
    os.system('mv -f %s %s' % (filename + '.png', IMAGES_DIRECTORY))
    plt.close()

def plot_trend(decomposed_series, index, point_id, days):
    '''
    Description: Methods to plot the stl components
    Parameters: statsmodel decomposed series object, datetime index, point_id and days
    '''
    trend = decomposed_series.trend.tolist()
    plt.figure(figsize=(70,42))
    plt.rcParams.update({'font.size': 40})
    plt.plot(index, trend, marker='', linestyle='-', label='Trend', linewidth=4)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.xticks(np.arange(0,len(index),step = len(index)/(days*4)), index[0::24], rotation=45)
    plt.grid(axis='both')
    plt.legend()
    plt.title("Trend of Point {0} over last {1} days".format(point_id, days))  
    plt.savefig("point{0}trend{1}".format(point_id, index[len(index)-1][:10]))

def plot_seasonal(decomposed_series, index, point_id, days):
    '''
    Description: Methods to plot the stl components
    Parameters: statsmodel decomposed series object, datetime index, point_id and days
    '''
    seasonal = decomposed_series.seasonal.tolist()
    plt.figure(figsize=(70,45))
    plt.rcParams.update({'font.size': 40})
    plt.plot(index[:96], seasonal[:96], marker='', linestyle='-', label='Seasonality', linewidth=4)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.xticks(np.arange(0,96,24), index[0:96:24], rotation=45)
    plt.grid(axis='both')
    plt.legend()
    plt.title("Seasonality of Point {0} over last {1} days".format(point_id, days))
    plt.savefig("point{0}seasonal{1}".format(point_id, index[len(index)-1][:10]))
 
def plot_residue(decomposed_series, index, point_id, days):
    '''
    Description: Methods to plot the stl components
    Parameters: statsmodel decomposed series object, datetime index, point_id and days
    '''
    resid = decomposed_series.resid.tolist()
    plt.figure(figsize=(70,45))
    plt.rcParams.update({'font.size': 40})
    plt.plot(index, resid, marker='', linestyle='-', label='Residue', linewidth=4)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.xticks(np.arange(0,len(index),step = len(index)/(days*4)), index[0::24], rotation=45)
    plt.grid(axis='both')
    plt.legend()
    plt.title("Residue of Point {0} over last {1} days".format(point_id, days))
    plt.savefig("point{0}resid{1}".format(point_id, index[len(index)-1][:10]))


def plot_components(decomposed_series, index, point_id, days):
    '''
    Description: Calls the three component plot methods
    Parameters: statsmodel object of decomposed series, datetime index array, point_id and days
    '''
    plot_trend(decomposed_series, index, point_id, days)
    plot_seasonal(decomposed_series, index, point_id, days)
    plot_residue(decomposed_series, index, point_id, days)

def main():
    #create_csv('Evans') - only needs to be done once per building
    evans_anomalies, evans_anom_counts = detect_evans_anomalies(plot_anom=True)
    evans_anomalies.transpose().to_csv('evans_anomalies.csv')
    evans_anom_counts.transpose().to_csv('evans_anom_counts.csv', index_label='Room')
    #stl_diff(583, 2233, 5,plot_ser=True, plot_comp=True)
    
    os.system('mv -f %s %s' % ('evans_anomalies.csv', CSV_DIRECTORY))
    os.system('mv -f %s %s' % ('evans_anom_counts.csv', CSV_DIRECTORY))

if __name__ == '__main__':
    main()


conn.close()

# stl.py

