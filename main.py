import math
import time
from datetime import datetime, timedelta
from heapq import nlargest

import boto3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyrebase
from boto3.dynamodb.conditions import Key
from shapely.geometry import LinearRing, Point, Polygon
from shapely.ops import nearest_points

import gdop
import kmeans
from config import *
from draw import draw
from fit_data import fit
from kalman_filter import KalmanFilter
from nls import nls
from particle_filter import create_gaussian_particles
from path_loss import log
from trilateration import *
from utils import *

dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')

firebase = pyrebase.initialize_app(FIREBASE)
db = firebase.database()
firebase_directory = FIREBASE['table']

history = {}
window_start = convert_date_to_secs(TRILATERATION['start'])


def run_kalman_filter_rss():

    # Query and plot unfiltered data
    t, y, avg = get_rss_fluctuation(
        RSS['start'], RSS['end'], RSS['ap'], RSS['mac'])
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
    """
    Runs localization for multiple mac devices 
    """
    global history
    dict_of_macs = {}

    if mode == "hist":
        data = get_hist_data()
        for _, mac in TRILATERATION['macs'].items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = compute_median_rss(data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = rss
            if dict_of_rss:
                dict_of_macs[mac] = dict_of_rss
        # print(dict_of_macs)

    elif mode == "live":
        data = get_live_data()
        # print(data)
        for _, mac in TRILATERATION['macs'].items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = get_live_rss_for_ap_and_mac_address(data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = rss
            if dict_of_rss:
                dict_of_macs[mac] = dict_of_rss

    elif mode == "replay":
        global window_start
        data = get_hist_data()
        for _, mac in TRILATERATION['macs'].items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = replay_hist_data(data, mac, ap['id'], window_start)
                if rss != -1:
                    dict_of_rss[ap['id']] = rss
            if dict_of_rss:
                dict_of_macs[mac] = dict_of_rss
        window_start += TRILATERATION['window_size']

    else:
        raise ValueError("Invalid run mode")

    for mac, dict_of_rss in dict_of_macs.items():
        r = {}
        for ap, rss in dict_of_rss.items():

            # Compute uncertainty
            if dict_of_rss:
                uncertainty = statistics.mean(
                    list(dict_of_rss.values())) * len(list(dict_of_rss.values()))
            uncertainty = abs(round(uncertainty/100, 1))
            # uncertainty = 0
            # print('Uncertainty: ', uncertainty)

            # Distance estimation
            if rss != -1 and rss > TRILATERATION['rss_threshold']:
                estimated_distance = log(rss)
                r[ap] = estimated_distance
                print('The estimated distance of the AP of RSS %d is %d is %f' %
                      (rss, ap, estimated_distance))

        # Points dictionary
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
        localization = None

        if len(p3) == 3:
            args = (p3[c[0]], p3[c[1]], p3[c[2]], r3[c[0]], r3[c[1]], r3[c[2]])
            estimated_localization = trilaterate(*args)
            print("Initial trilateration estimate: ", estimated_localization)

            # Using APs with highest GDOP for trilateration
            try:
                loc = nls(estimated_localization, p, r)
                gdops = gdop.compute_all(loc, r)
                min_gdop = gdop.get_best_combination(gdops)
                c = min_gdop[0]
                # print(gdops)
                print("Minimum GDoP: ", c)
                p3 = {k: v for k, v in p.items() if k in c}
                r3 = {k: v for k, v in r.items() if k in c}
                args = (p3[c[0]], p3[c[1]], p3[c[2]],
                        r3[c[0]], r3[c[1]], r3[c[2]])
                estimated_localization = trilaterate(*args)
                print("New trilateration estimate: ", estimated_localization)
            except np.linalg.LinAlgError:
                pass

            # Non-linear least squares
            localization = nls(estimated_localization, p, r)
            print("NLS estimate: ", tuple(localization[:2]))

            # Correct angle deviation
            localization = rotate(localization, GEO['deviation'])
            user = list(TRILATERATION['macs'].keys())[
                list(TRILATERATION['macs'].values()).index(mac)]
            print("Localization of %s is %s" % (user, localization))

            # Draw
            # draw(estimated_localization, localization, p, r)

            # Compute absolute localization
            lat = GEO['origin'][0] + localization[1]*GEO['oneMeterLat']
            lng = GEO['origin'][1] + localization[0]*GEO['oneMeterLng']

            # Move invalid point inside building
            polygon = Polygon(BUILDING)
            point = Point(lat, lng)
            if not polygon.contains(point):
                p1, _ = nearest_points(polygon, point)
                lat, lng = p1.x, p1.y
            print("Physical location: ", (lat, lng))

            # Save localization history
            history.setdefault(user, []).append((lat, lng))

            # Push data to Firebase
            if mode == 'live':
                data = {
                    'mac': mac,
                    'lat': lat,
                    'lng': lng,
                    'radius': str(uncertainty),
                    'timestamp': str(datetime.now())
                }
                db.child(firebase_directory).push(data)

        elif localization != None:
            print("info: trilateration not possible, using last value ", localization)


def main():

    # Mode 1: Trilateration on historical data
    # run("hist")

    # Mode 2: Trilateration in real-time
    # while(True):
    #     run("live")

    # Mode 3: Replay historical data
    window_end = convert_date_to_secs(TRILATERATION['end'])
    for _ in range(window_start, window_end, TRILATERATION['window_size']):
        run("replay")
    print(history)
    find_in_building(history)

    # Fit curve
    # fit()

    # Kalman filter
    # run_kalman_filter_rss()

    # Particle filter
    # print(create_gaussian_particles([0, 1, 2], [0, 1, 2], 1000))

    # gdops = gdop.compute_all((1, 1), {'21': 2, '55': 5, '56': 9, '57': 6})
    # gdop.compute((1, 1), {'21': 2, '55': 5, '56': 9, '57': 6})
    # c = gdop.get_best_combination(gdops)
    # print(c)

    # kmeans.cluster()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
