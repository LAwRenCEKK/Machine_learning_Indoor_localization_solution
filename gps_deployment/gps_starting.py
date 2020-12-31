"""
Author: Jianpeng Liu
Copyrights belong to WiSeR Lab 
Downstream Localiztion model GPs
Reference: https://rse-lab.cs.washington.edu/postscripts/gp-localization-rss-06.pdf
"""
import json
import math
import numpy as np
import pandas as pd
from gps_deployment.main_controller import MainController


def gps_fromxy(lat0,lon0,x_offset,y_offset):
    """ Convert back to the GPS cooridnate system from the local system.
    Args:
        lat0,lon0: latitude and longitude of the reference point(origin)
        x_offset,y_offset: The offset in X-axis and Y-axis
    Returns:
        GPS cooridnate(longitude and latitude)
    """
    R = 6378137.0
    #offset in radians
    dlat = y_offset/R
    dlon = x_offset/(R * math.cos(math.pi * lat0 /180.0))
    #offset gps in decimal degrees
    lat1 = lat0 + dlat * 180/math.pi
    lon1 = lon0 + dlon * 180/math.pi
    return [lon1,lat1]


def gps_start(num_ap, training_data):
    """ GPs models training
    Args:
        num_ap: Number of GPs to initialize
        training_data: training_data for each GPs
    Returns:
        GPs model objects
    """
    mc = MainController(num_ap=num_ap, training_data = training_data)
    mc.load_data()
    mc.initialize_gaussian_processes_optimal(visualize=True)
    for ap_index in range(0,mc.number_of_access_points):
        mc.visualize_gaussian(ap_index, mc.gaussian_processes[ap_index])
    return mc


def gps_evaluation_uji(gps,num_ap,test_data,lat_min,lat_max,log_min,log_max,l):
    """ GPs models performance evaluation for UJI building
    Args:
        gps: GPs model objects
        num_ap: used to locate the specific GPs model
        test_data: test data for performance evaluation
        lat_min,lat_max,log_min,log_max: specify the boundary of the building 
    Returns:
        Evaluation result 
    """
    sum_loc_xy = []
    for ts_data in test_data:
        maximum = 0
        loc_xy = [0,0]
        X_range = np.arange(lat_min,lat_max,0.05)
        Y_range = np.arange(log_min,log_max,0.05)
        for i in range(0,len(X_range)):
            for j in range(0,len(Y_range)):
                prod_maximum = 1
                for k in range(num_ap):
                    x = X_range[i]
                    y = Y_range[j]
                    temp = gps.gaussian_processes[k].predict_gaussian_value2([x,y],ts_data[k])
                    try:
                        prod_maximum = prod_maximum*temp
                    except:
                        pass
                if maximum < prod_maximum:
                    maximum = prod_maximum
                    loc_xy = [x,y]

        sum_loc_xy.append(loc_xy)

    errors_xy = test_data[:,-2:] - sum_loc_xy
    rlt_xy = np.median(np.sqrt(abs(errors_xy[:,0])**2 + abs(errors_xy[:,1])**2))
    print ("GPs error: "+ str(rlt_xy))

    return {"x-y": rlt_xy}


def gps_evaluation(gps, num_ap, test_data,l):
    """ GPs models performance evaluation for MDCC buildings
    Args:
        gps: GPs model objects
        num_ap: used to locate the specific GPs model
        test_data: test data for performance evaluation
    Returns:
        Evaluation result 
    """
    sum_loc_xy = []
    for ts_data in test_data:
        maximum = np.NINF
        loc_xy = [0,0]
        X_range = np.arange(0,100,5)
        Y_range = np.arange(0,100,5)
        for i in range(0,len(X_range)):
            for j in range(0,len(Y_range)):
                prod_maximum = 1
                for k in range(num_ap):
                    x = X_range[i]
                    y = Y_range[j]
                    try:
                        temp = np.log(gps.gaussian_processes[k].predict_gaussian_value2([x,y],ts_data[k]))
                        prod_maximum = prod_maximum*temp
                    except:
                        pass
                
                if maximum < prod_maximum:
                    maximum = prod_maximum
                    loc_xy = [x,y]
        sum_loc_xy.append(loc_xy)
    
    errors_xy = test_data[:,-2:] - sum_loc_xy
    rlt_xy = np.median(np.sqrt(errors_xy[:,0]**2 + errors_xy[:,1]**2))
    
    print ("GPs error: "+ str(rlt_xy))

    return {"x-y": rlt_xy}


def estimate(gps, num_ap, test_data):
    """ GPs models performance evaluation for MDCC buildings
    Args:
        gps: GPs model objects
        num_ap: used to locate the specific GPs model
        test_data: test data
    Returns:
        Estimate result 
    """
    maximum = np.NINF
    loc_xy = [0,0]
    X_range = np.arange(20,40,2)
    Y_range = np.arange(20,40,2)
    for i in range(0,len(X_range)):
        for j in range(0,len(Y_range)):
            prod_maximum = 1
            for k in range(num_ap):
                x = X_range[i]
                y = Y_range[j]
                try:
                    tem = gps.gaussian_processes[k].predict_gaussian_value2([x,y],test_data[k])
                    prod_maximum = prod_maximum*tem
                except Exception as e:
                    pass
            if maximum < prod_maximum:
                maximum = prod_maximum
                loc_xy = [x,y]

    return {"location":loc_xy}
