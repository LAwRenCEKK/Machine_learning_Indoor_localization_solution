import math,glob,os
import Data_pb2
import numpy as np

def distance(lat1,lon1,lat2,lon2):
    """Distance between two GPS point
    Args:
        - lat1,lon1: latitude and longitude of the first GPS location
        - lat2,lon2: latitude and longitude of the second GPS location
    Returns:
        Distance between two GPS point in meters
    """
    radius = 6378.137 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
          * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d*1000

def offset_coord(lat0,lon0,lat1,lon1):
    """Compute the offset of the seond GPS point relative to the first point
    in world coordinate system
    Args:
        lat0,lon0 -- the latitude and longittude of the first GPS point
        lat1,lon1 -- the latitude and longittude of the second GPS point

    Returns:
        offset coordinate in meters (x_offset,y_offset)
    """
    x_project = (lat0,lon1)
    y_project = (lat1,lon0)
    x_offset = distance(lat0,lon0,x_project[0],x_project[1])
    y_offset = distance(lat0,lon0,y_project[0],y_project[1])
    #if at the south of then flip the sign
    if lat1 < lat0:
        y_offset = y_offset * (-1)
    #if at the west of then flip the sign
    if lon1 < lon0:
        x_offset = x_offset * (-1)
    return (x_offset, y_offset)

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
    return (lon1,lat1)


def gps2local(gps_origin,gps_locations):
    """Convert rss of GPS cooridnate into local coordinate system
    Args:
        gps_origin: the choosen gps origin point
        gps_locations: a series of locations in gps format
    Returns:
    cooridnate in a local coordinate system
    """
    loc_coords = []
    for gps_loc in gps_locations:
        local_coord = offset_coord(gps_origin[1],gps_origin[0],gps_loc[1],gps_loc[0])
        loc_coords.append([local_coord[0],local_coord[1]])
    return loc_coords

def get_file_list(folder, mode):
    """Get fingerprint file list inside a folder
    Args:
        folder - folder of fingerprint files
        mode - collect mode, 1: path based, 2: point-based
    Returns:
        List of filenames of the specified files inside the folder
    """
    all_files = glob.glob(folder+"/*.pbf")
    fp_files = []
    for fname in all_files:
        parts = os.path.basename(fname).split('_')
        if len(parts) == 3 and int(parts[1]) == mode:
            fp_files.append(fname)
    #Replace with full paths
    return [os.path.realpath(f) for f in fp_files]

def load_data_packs(filelist):
    """Load the data packages from the file list
    Args:
        filelist - a list of file packs
    Returns:
        a list of data packages
    """
    datapacks = []
    for f in filelist:
        datapack = Data_pb2.DataPack()
        with open(f,'rb') as fin:
            datapack.ParseFromString(fin.read())
            datapacks.append(datapack)
    return datapacks

def location_interpolate(start, terminal, speed, time):
    """Interpolate the location during walking

    Args:
        start: The start location, type: Data_pb2.LatLon (GPS coordinate)
        terminal: The terminal location, type: Data_pb2.LatLon (GPS coordinate)
        start_time: The start time of walking
        terminal_time: The termial time of walking
        speed: the walk speed
        time: the walk time

    Returns:
        (latitude,longitude) <GPS coordinate>
    """
    start_latitude = start.latitude
    start_longitude = start.longitude
    terminal_latitude = terminal.latitude
    terminal_longitude = terminal.longitude
    pathlength = distance(start_latitude,start_longitude,
                                   terminal_latitude,terminal_longitude)
    #Interpolate
    walklength = speed * time
    ratio = walklength/pathlength
    total_xoffset, total_yoffset = offset_coord(start_latitude,\
                        start_longitude,terminal_latitude,terminal_longitude)
    #Note the return of gps_fromxy is lon,lat
    reslon,reslat = gps_fromxy(start_latitude,start_longitude,\
                                   ratio * total_xoffset,ratio * total_yoffset)

    return (reslon,reslat)

def parse_path_packages(path_packages,AP_bssids_index_dict):
    """Parse the WiFi signal values in packages
    Args:
        path_packages: a list of packages
        AP_bssids_index_dict: the dict of AP
    Returns:
        (M,L), M is the data matrix(2D array), L is the loation labels
    """
    #Create the fingerprint and location label matrix.
    FPs = []
    labels = []
    #For the path data, to simplify, three assumptions are made
    #1. walk with the same speed. 2.path is straight line. 3. the scan time is the same for all bssids.
    for datapack in path_packages:
        walk_time = (datapack.terminalTime - datapack.startTime)/1000.0
        walk_distance = distance(datapack.startLocation.latitude,datapack.startLocation.longitude,\
                                 datapack.terminalLocation.latitude,datapack.terminalLocation.longitude)
        print('path length is %f meter' % walk_distance)
        walk_speed = walk_distance/walk_time
        #estimate the locations for each scan
        scan_seq = 0
        for e in datapack.rssItems:
            if e.scanNum == scan_seq:
                if e.bssid in AP_bssids_index_dict:
                    fp[AP_bssids_index_dict[e.bssid]] = e.level * 1.0
                    bssid_timestamp[e.bssid]=e.timestamp
            else:
                #new scan encountered, save the previous scan and re-init
                if scan_seq!=0:
                    #Remove out liers from fp by using the bssid_timestamp
                    for k,v in bssid_timestamp.items():
                        if np.abs(v - np.median(list(bssid_timestamp.values())))>1000:
                            fp[AP_bssids_index_dict[k]]=-93
                            #print('outliers from AP',k)
                    FPs.append(fp)
                    t = np.median(list(bssid_timestamp.values()))/1000.0 + datapack.deviceBootTime - datapack.startTime
                    walktime = t/1000.0
                    loc = location_interpolate(datapack.startLocation, datapack.terminalLocation, walk_speed, walktime)
                    labels.append(loc)
                    #print("Scan ID",scan_seq,",duration",scan_duration)
                #init and record the first of new scan
                scan_seq = e.scanNum
                fp = [-93.0] * len(AP_bssids_index_dict)
                bssid_timestamp = {}
                if e.bssid in AP_bssids_index_dict:
                    fp[AP_bssids_index_dict[e.bssid]] = e.level * 1.0
                    bssid_timestamp[e.bssid]=e.timestamp
        #Add the last scan
        for k,v in bssid_timestamp.items():
            if np.abs(v - np.median(list(bssid_timestamp.values())))>1000:
                fp[AP_bssids_index_dict[k]]=-93
                #print('outliers from AP',k)

        FPs.append(fp)
        t = np.median(list(bssid_timestamp.values()))/1000.0 + datapack.deviceBootTime - datapack.startTime
        walktime = t/1000.0
        #print("Scan ID",scan_seq,",duration",scan_duration)
        loc = location_interpolate(datapack.startLocation, datapack.terminalLocation, walk_speed, walktime)
        labels.append(loc)
    #end for
    return (FPs,labels)

def parse_point_packages(point_packages,AP_bssids_index_dict):
    """Parse point packages
    """
    point_FPs = []
    point_labels = []
    #Load the point collected data
    for datapack in point_packages:
        loc = (datapack.startLocation.longitude,datapack.startLocation.latitude)
        scan_seq = 0
        fp_scans = []
        for e in datapack.rssItems:
            if e.scanNum == scan_seq:
                if e.bssid in AP_bssids_index_dict:
                    fp[AP_bssids_index_dict[e.bssid]] = e.level * 1.0
                    bssid_timestamp[e.bssid]=e.timestamp
            else:
                #new scan encountered, save the previous scan and re-init
                if scan_seq!=0:
                    #Remove out liers from fp by using the bssid_timestamp
                    for k,v in bssid_timestamp.items():
                        if np.abs(v - np.median(list(bssid_timestamp.values())))>1000:
                            fp[AP_bssids_index_dict[k]]=-93
                            #print('outliers from AP',k)
                    fp_scans.append(fp)
                    #print("Scan ID",scan_seq,",duration",scan_duration)

                #init and record the first of new scan
                scan_seq = e.scanNum
                fp = [-93.0] * len(AP_bssids_index_dict)
                bssid_timestamp = {}

                if e.bssid in AP_bssids_index_dict:
                    fp[AP_bssids_index_dict[e.bssid]] = e.level * 1.0
                    bssid_timestamp[e.bssid]=e.timestamp
        #Add the last scan
        for k,v in bssid_timestamp.items():
            if np.abs(v - np.median(list(bssid_timestamp.values())))>1000:
                fp[AP_bssids_index_dict[k]]=-93
                #print('outliers from AP',k)
        fp_scans.append(fp)
        point_FPs.append(np.mean(fp_scans,axis=0).tolist())
        point_labels.append(loc)
    #end for
    return (point_FPs,point_labels)
