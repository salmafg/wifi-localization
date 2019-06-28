import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
import time
import json
import numpy as np
import math
from config import *
from draw import draw
from utils import *
from heapq import nlargest
from path_loss import log
import matplotlib
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from nls import nls
from trilateration import *
import pyrebase
from fit_data import fit


dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')


def run_kalman_filter_rss():

    # Query and plot unfiltered data
    t, y, avg = get_rss_fluctuation_by_mac_address(
        RSS['start'], RSS['end'], RSS['ap'])
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


def run(mode):

    # Query RSS data by mac address
    dict_of_rss = get_data_by_mac_address(
        mode, TRILATERATION['mac'], TRILATERATION['aps'])

    # Distance estimates dictionary
    global r
    r = {}
    for ap, rss in dict_of_rss.items():
        if rss > -60:  # threshold
            estimated_distance = log(rss)
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
        print("Corrected localization: ", localization)

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
    # run_kalman_filter_rss(29)

    # Mode 1: Trilateration on historical data
    # run("hist")

    # Mode 2: Trilateration in real-time
    while(True):
        run("live")

    # Fit curve
    # fit()


if __name__ == "__main__":
    main()
