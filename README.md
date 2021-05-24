# data-analysis
Data analysis files from the server

## STL.py

This function implements Seasonal Trend Loess Decomposition using pandas and the stldecompose library, as well as basic heuristic models, for anomaly detection. It is run daily by user energy's crontab, generating the anomaly counts of different types for Evans, storing those csv files in  /var/www/frontend.static/csv-files. It also generates graphs of the different data points for each room, and stores them in /var/www/frontend.static/images. 

## To Add a new building to the file:

There are three important steps to use the existing code on a new building:
1. Create a csv file of all the rooms and corresponding points in the building using a create_buildingname_csv() method. 
2. write detect_buildingname_anomalies and detect_buildingname_room methods. 
3. call the detect_buildingname_anomalies method from the main method, and return the result to the correct directories. 

To do step...
1: Thanks to hard work from Silas and the magic of Brick, this step is now super easy. Be grateful to Silas, he saved you a good several hours of work. 

All you have to do is: change the line in the main method to run create_csv() with the building you want to use. Make sure it is capitalized properly the way Evans is. 

2:You can essentially just copy over both detect_evans_anomalies and detect_evans_room, and then just change the strings so they are appropriate for the points you are using, and that you are calling in the csv file for the correct building. The comments in the code explain clearly what is going on in these methods, so read those to be sure. These methods are the bulk of the building-specific code and the rest of the methods are intended to be widely used for any building. (NOTE: the temp_anomalies, valve_anomalies, and virtset_temp_diff_anomalies are more customized versions for each point in evans, but these methods could be used for those types of points in any building. If you have other data points that don't fit those, you can either write another more customized method, or use the stl_point_anomalies, which has an optional second parameter, and is more generalized to run stl on any point, but not the heuristics.)

3. This part is easy, just go to the main method and echo what is being done with Evans to run the detect_buildingname_anomalies method and move the csv files to where you want them. Lines 35-37 are where the global variables for the directory locations are, so if you change where you want those, make sure to update the variables accordingly. The graphs are generated by plot_anomalies, and saved to the IMAGE_DIRECTORY within there. 
