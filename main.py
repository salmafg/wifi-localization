import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
import time
import json
import numpy as np
import math
import statistics
from config import *
from draw import draw
from utils import rotate
from heapq import nlargest
from path_loss import log
import matplotlib
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from statistics import stdev
from nls import nls
from trilateration import *
import pyrebase


dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')


def get_rss_fluctuation_by_mac_address(start, end, mac, ap):
    start_timestamp = datetime.strptime(start, '%d %b %Y %H:%M')
    end_timestamp = datetime.strptime(end, '%d %b %Y %H:%M')

    start_in_sec = int(start_timestamp.timestamp())
    end_in_sec = int(end_timestamp.timestamp())

    response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap)
                              & Key('timestamp').between(start_in_sec, end_in_sec))

    avg_rss = compute_avg_rss_for_mac_address(response, mac)
    rss = []
    timestamps = []
    for r in response['Items']:
        if (r['payload']['mac']) == mac:
            rss.append(int(r['payload']['rssi']))
            timestamps.append(int(r['payload']['timestamp']))
    return(timestamps, rss, avg_rss)


def run_kalman_filter_rss():

    # Query and plot unfiltered data
    t, y, avg = get_rss_fluctuation_by_mac_address(
        RSS['start'], RSS['end'], RSS['mac'], RSS['ap'])
    devs = [-int(abs(i)-abs(avg)) if i <= avg else -
            int(abs(avg)-abs(i)) for i in y]
    plt.hist(devs, color='blue', edgecolor='black', bins=int(len(y)/10))
    plt.xlim(left=min(y))
    plt.show()
    tperiod = max(t) - min(t)
    trange = np.linspace(0, tperiod, len(y))
    print('Avg. RSS: ', avg)
    plt.plot(trange, y)
    plt.hlines(avg, 0, tperiod, 'k', 'dashed')
    plt.title('RSS fluctuation')
    plt.show()

    plt.hist(y, color='blue', edgecolor='black', bins=int(len(t)/10))
    plt.title('Histogram of RSS at 1m')
    plt.xlabel('RSS')
    plt.show()

    # Apply Kalman filter and plot results
    kalman = KalmanFilter(0.008, 0.1)
    filtered_data = []
    sum_filtered = 0
    for i in y:
        filtered_i = kalman.filter(i)
        sum_filtered += filtered_i
        filtered_data.append(filtered_i)
    avg_filtered = sum_filtered / len(filtered_data)
    print('Avg. filtered RSS: ', avg_filtered)
    plt.plot(trange, filtered_data)
    plt.ylim(min(y), max(y))
    plt.hlines(avg_filtered, 0, tperiod, 'k', 'dashed')
    plt.show()

    plt.hist(filtered_data, color='blue',
             edgecolor='black', bins=int(len(t)/10))
    plt.title('Histogram of filtered RSS at 1m')
    plt.xlim(min(y), max(y))
    plt.xlabel('RSS')
    plt.show()


def get_data_by_mac_address(mode, mac, APs):
    """
    Query data from AWS using mac address
    """
    dict_of_processed_rss = {}

    for ap in APs:

        # Query historical data in a specified time range
        if mode == "hist":
            start_timestamp = datetime.strptime(
                TRILATERATION['start'], '%d %b %Y %H:%M')
            end_timestamp = datetime.strptime(
                TRILATERATION['end'], '%d %b %Y %H:%M')

            start_in_sec = int(start_timestamp.timestamp())
            end_in_sec = int(end_timestamp.timestamp())

            response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap['id'])
                                      & Key('timestamp').between(start_in_sec, end_in_sec))

            # rss = compute_avg_rss_for_mac_address(response, mac)
            rss = compute_median_rss_for_mac_address(response, mac)
            dict_of_processed_rss[ap['id']] = rss

        # Query real-time data
        elif mode == "live":
            now_in_sec = int(datetime.now().timestamp())
            response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap['id'])
                                      & Key('timestamp').gte(now_in_sec-5))
            rss = get_live_rss_for_mac_address(response, mac)

            if rss == -1:
                print("warning: no live data detected for AP", ap['id'])
            else:
                dict_of_processed_rss[ap['id']] = rss

    return dict_of_processed_rss


def compute_avg_rss_for_mac_address(response, mac):
    sum_rss = 0
    count_rss = 0
    for r in response['Items']:
        if (r['payload']['mac']) == mac:
            # print(r['payload'])
            sum_rss = sum_rss + r['payload']['rssi']
            count_rss += 1
    avg_rss = sum_rss / count_rss
    # print(avg_rss)
    return avg_rss


def compute_median_rss_for_mac_address(response, mac):
    rss_values = []
    for r in response['Items']:
        if (r['payload']['mac']) == mac:
            # print(r['payload'])
            rss_values.append(r['payload']['rssi'])
    median_rss = statistics.median(rss_values)
    # print(median_rss)
    return median_rss


def get_live_rss_for_mac_address(response, mac):
    for r in response['Items']:
        if (r['payload']['mac']) == mac:
            return r['payload']['rssi']
    return -1


def run(mode):

    # Query RSS data by mac address
    dict_of_rss = get_data_by_mac_address(
        mode, TRILATERATION['mac'], TRILATERATION['aps'])

    # Distance estimates dictionary
    global r
    r = {}
    for ap, rss in dict_of_rss.items():
        if rss > -60:
            estimated_distance = log(rss, 2.3)
            # estimated_distance = fspl(rss)
            r[ap] = estimated_distance
            print('The estimated distance of the AP for RSS %d is %d is %f' %
                  (rss, ap, estimated_distance))

    # Points dictionary
    global p
    p = {}
    for i in r:
        p[i] = next(item for item in TRILATERATION['aps']
                    if item['id'] == i)['xy']

    c = sorted(r, key=r.get)[:3]
    if c:
        print("Closest to access points", ', '.join(str(i) for i in c))

    # Trilateration
    p3 = {k: v for k, v in p.items() if k in c}
    r3 = {k: v for k, v in r.items() if k in c}
    global localization
    localization = None

    if len(p3) == 3:
        args = (p3[c[0]], p3[c[1]], p3[c[2]], r3[c[0]], r3[c[1]], r3[c[2]])
        estimated_localization = trilaterate(*args)
        print("Trilateration estimation: ", estimated_localization)

        # Non-linear least squares
        localization = nls(estimated_localization, p, r)
        print("NLS estimation: ", tuple(localization[:2]))

        # Correct angle deviation
        localization = rotate(localization, GEO['deviation'])

        # Draw
        # draw(estimated_localization, localization, p, r)

        # Connect Firebase
        firebase = pyrebase.initialize_app(FIREBASE)
        db = firebase.database()
        firebase_directory = "trilateration-data"

        # Compute absolute localization
        lat = GEO['origin'][0] + localization[1]*GEO['oneMeterLat']
        lng = GEO['origin'][1] + localization[0]*GEO['oneMeterLng']

        # Push data to Firebase
        data = {
            'lat': lat,
            'lng': lng,
            'timestamp': str(datetime.now())
        }
        db.child(firebase_directory).push(data)

    elif localization != None:
        print("info: trilateration not possible, using last value ", localization)


def main():

    # Kalman filter
    # run_kalman_filter_rss()

    # Mode 1: Trilateration on historical data
    # run("hist")

    # Mode 2: Trilateration in real-time
    while(True):
        run("live")
        time.sleep(1)


if __name__ == "__main__":
    main()
