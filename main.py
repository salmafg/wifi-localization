import csv
import json
import math
import random
import string
import subprocess
import time
from datetime import datetime, timedelta
from heapq import nlargest

import boto3
import gmplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyrebase
from boto3.dynamodb.conditions import Key
from shapely.geometry import LinearRing, Point, Polygon
from shapely.ops import nearest_points

import gdop
import hmm
import kmeans
from config import *
from draw import draw
from fit_data import fit
from kalman_filter import KalmanFilter
from nls import nls
from path_loss import log
from trilateration import *
from utils import *

dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')

firebase = pyrebase.initialize_app(FIREBASE)
db = firebase.database()
firebase_table = FIREBASE['table']

dict_of_macs = TRILATERATION['macs']
window_start = convert_date_to_secs(TRILATERATION['start'])
geo_history = {}
sem_history = json.loads(open('data.json').read())
hmm_predictions = {}
model = None

with open('usernames.csv', 'r') as f:
    reader = csv.reader(f)
    usernames = flatten(list(reader))


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


def run(mode, data=None):
    '''
    Runs localization for multiple mac devices 
    '''
    global usernames
    global window_start
    dict_of_mac_rss = {}

    if mode == 'hist':
        for mac, _ in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = compute_median_rss(data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = rss
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
        # print(dict_of_mac_rss)

    elif mode == 'live':
        data = get_live_data()
        for mac, _ in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = get_live_rss_for_ap_and_mac_address(data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = rss
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss

    elif mode == 'live-all':
        data = get_live_data()
        for r in data:
            if r['payload']['mac'] not in dict_of_macs:
                random.shuffle(usernames)
                username = usernames.pop()
                dict_of_macs[r['payload']['mac']] = username
        for mac, _ in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = get_live_rss_for_ap_and_mac_address(data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = rss
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss

    elif mode == 'replay':
        day = datetime.fromtimestamp(window_start).strftime('%A')
        time = datetime.fromtimestamp(
            window_start).strftime('%H:%M:%S')
        for mac, _ in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss, data = replay_hist_data(data, mac, ap['id'], window_start)
                if rss != -1:
                    dict_of_rss[ap['id']] = rss
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
        window_start += TRILATERATION['window_size']

    else:
        raise ValueError('error: invalid run mode')

    for mac, dict_of_rss in dict_of_mac_rss.items():
        r = {}
        for ap, rss in dict_of_rss.items():

            # Distance estimation
            user = list(dict_of_macs.values())[
                list(dict_of_macs.keys()).index(mac)]
            if rss != -1 and rss > TRILATERATION['rss_threshold']:
                estimated_distance = log(rss)
                r[ap] = estimated_distance
                print('The estimated distance of %s from AP %d of RSS %d is %f' %
                      (user, ap, rss, estimated_distance))

        # Points dictionary
        p = {}
        for i in r:
            p[i] = next(item for item in TRILATERATION['aps']
                        if item['id'] == i)['xy']

        c = sorted(r, key=r.get)[:3]
        if c:
            print('Closest to access points:', ', '.join(str(i) for i in c))

        # Trilateration
        p3 = {k: v for k, v in p.items() if k in c}
        r3 = {k: v for k, v in r.items() if k in c}
        localization = None

        if len(p3) == 3:
            args = (p3[c[0]], p3[c[1]], p3[c[2]], r3[c[0]], r3[c[1]], r3[c[2]])
            estimated_localization = trilaterate(*args)
            print('Initial trilateration estimate:', estimated_localization)

            # Compute uncertainty
            try:
                loc = nls(estimated_localization, p, r)
                gdops = gdop.compute_all(loc, r)
                min_gdop = gdop.get_best_combination(gdops)
                if min_gdop[1] > TRILATERATION['max_uncertainty']:
                    uncertainty = TRILATERATION['max_uncertainty']
                elif min_gdop[1] < TRILATERATION['min_uncertainty']:
                    uncertainty = TRILATERATION['min_uncertainty']
                else:
                    uncertainty = round(min_gdop[1], 1)
            except np.linalg.LinAlgError:
                uncertainty = TRILATERATION['default_uncertainty']
            print('Uncertainty: %dm' % uncertainty)

            # Non-linear least squares
            localization = nls(estimated_localization, p, r)
            print('NLS estimate:', tuple(localization[:2]))

            # Correct angle deviation
            localization = rotate(localization, GEO['deviation'])
            user = list(dict_of_macs.values())[
                list(dict_of_macs.keys()).index(mac)]
            print('Relative location:', localization)

            # Draw
            # draw(estimated_localization, localization, p, r)

            # Compute absolute localization
            lat = GEO['origin'][0] + localization[1]*GEO['oneMeterLat']
            lng = GEO['origin'][1] + localization[0]*GEO['oneMeterLng']

            # Move invalid point inside building to a valid location
            room = get_room_by_physical_location(lat, lng)
            if room is None:
                closest_polygon, closest_room = get_closest_polygon(lng, lat)
                point = Point(lng, lat)
                p1, _ = nearest_points(closest_polygon, point)
                d = point.distance(p1)
                lng, lat = p1.x, p1.y
                room = closest_room
                print('...point was moved %fm' % d)

            # Write to file if uncertainty is not too high
            if uncertainty < TRILATERATION['max_uncertainty']:
                geo_history.setdefault(user, []).append((lat, lng))
                sem_history.setdefault(user, []).append(room)
                if mode == 'live':
                    data = json.dumps(sem_history)
                    f = open("data.json", "w")
                    f.write(data)
                    f.close()

            # Print observation
            if mode != 'live':
                print('>> %s was observed in %s on %s %s' %
                      (user, room, day, time))
            else:
                print('>> %s was just observed in %s' %
                      (user, room))
            print('Physical location:', (lat, lng))

            # Push data to Firebase
            data = {
                'user': user,
                'mac': mac,
                'lat': lat,
                'lng': lng,
                'radius': str(uncertainty),
                'timestamp': str(datetime.now())
            }
            db.child(firebase_table).push(data)

        elif localization != None:
            print('info: trilateration not possible, using last value', localization)


def main():

    # Mode 1: Trilateration on historical data
    # data = get_hist_data()
    # run('hist', data)

    # Mode 2: Trilateration in real-time
    # while(True):
    #     run('live', None)

    # Mode 3: Replay historical data and parse observations to json
    data = get_hist_data()
    print('Data retrieved.')
    global usernames
    for r in data:
        if r['payload']['mac'] not in dict_of_macs:
            if usernames:
                random.shuffle(usernames)
                username = usernames.pop()
            else:
                username = 'user'+''.join(random.choices(string.digits, k=3))
            dict_of_macs[r['payload']['mac']] = username
    window_end = convert_date_to_secs(TRILATERATION['end'])
    for _ in range(window_start, window_end, TRILATERATION['window_size']):
        run('replay', data)
    plot_localization(sem_history)

    # Fit HMM from JSON and make predications
    # global model
    # obs = json.loads(open('data.json').read())
    # model = hmm.fit(obs)
    # hmm.predict_all(model, obs, 'map')

    # Fit curve
    # fit()

    # Kalman filter
    # run_kalman_filter_rss()

    # kmeans.cluster()


if __name__ == '__main__':
    try:
        main()
        subprocess.Popen(['notify-send', "Localization complete."])
    except KeyboardInterrupt:
        pass
