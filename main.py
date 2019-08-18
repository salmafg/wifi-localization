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

import classifier
import gdop
import hmm
import kmeans
import mqtt
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
rel_hist = {}
geo_hist = {}
sem_hist = json.loads(open('locations.json').read())
rss_hist = json.loads(open('rss.json').read())
y_true = []
y_pred = []


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


def run(mode, data=None, model=None):
    '''
    Runs localization for multiple mac devices 
    '''
    global usernames
    global window_start
    global rss_hist
    dict_of_mac_rss = {}

    if mode == 'live':
        data = get_live_data()
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = get_live_rss_for_ap_and_mac_address(data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
            if len(dict_of_rss) >= 3:
                rss_hist.setdefault(user, []).append(dict_of_rss)

    if mode == 'mqtt':
        data = mqtt.get_messages()
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = mqtt_get_live_rss_for_ap_and_mac_address(
                    data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
            if len(dict_of_rss) >= 3:
                rss_hist.setdefault(user, []).append(dict_of_rss)
        # print(data)

    elif mode == 'live-all':
        data = get_live_data()
        for r in data:
            if r['payload']['mac'] not in dict_of_macs:
                if usernames:
                    random.shuffle(usernames)
                    username = usernames.pop()
                else:
                    username = 'user' + \
                        ''.join(random.choices(string.digits, k=3))
                dict_of_macs[r['payload']['mac']] = username
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = get_live_rss_for_ap_and_mac_address(data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
            if len(dict_of_rss) >= 3:
                rss_hist.setdefault(user, []).append(dict_of_rss)

    elif mode == 'mqtt-all':
        data = mqtt.get_messages()
        for r in data:
            r = json.loads(r)
            if r['mac'] not in dict_of_macs:
                if usernames:
                    random.shuffle(usernames)
                    username = usernames.pop()
                else:
                    username = 'user' + \
                        ''.join(random.choices(string.digits, k=3))
                dict_of_macs[r['mac']] = username
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = mqtt_get_live_rss_for_ap_and_mac_address(
                    data, mac, ap['id'])
                if rss != -1:
                    dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
            if len(dict_of_rss) >= 3:
                rss_hist.setdefault(user, []).append(dict_of_rss)

    elif mode == 'replay':
        day = datetime.fromtimestamp(window_start).strftime('%A')
        time = datetime.fromtimestamp(
            window_start).strftime('%H:%M:%S')
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss, data = replay_hist_data(data, mac, ap['id'], window_start)
                if rss != -1 and rss > TRILATERATION['rss_threshold']:
                    dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
            if len(dict_of_rss) >= 3:
                rss_hist.setdefault(user, []).append(dict_of_rss)
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
            uncertainty = min(r.values())
            try:
                loc = nls(estimated_localization, p, r)
                d_1 = distance(loc, rel_hist[user][len(rel_hist[user])-1])
                if d_1 <= TRILATERATION['default_uncertainty']:
                    # uncertainty = min(r.values()) + d_1
                    uncertainty = max(min(r.values()), d_1)
                else:
                    try:
                        d_2 = distance(
                            loc, rel_hist[user][len(rel_hist[user])-2])
                        if d_2 <= TRILATERATION['default_uncertainty']:
                            # uncertainty = min(r.values()) + d_2
                            uncertainty = max(min(r.values()), d_2)
                        else:
                            # uncertainty = min(r.values()) + d_1
                            uncertainty = max(min(r.values()), d_1)
                    except:
                        pass
            except:
                pass
            print('Uncertainty: %.1fm' % round(uncertainty, 1))

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

            # Machine learning prediction
            if model is not None:
                temp = {}
                temp[mac] = [dict_of_mac_rss[mac]]
                temp, _ = tidy_rss(temp)
                temp = np.atleast_2d(temp)
                pred, prob = classifier.predict_room(model, temp)
                print('>> model prediction in %s with probability %.1f' %
                      (STATES[pred], round(prob, 1)))
                if prob >= ML['prob_threshold'] and room != pred:
                    point = Point(lng, lat)
                    pred_polygon = Polygon(
                        MAP[pred]['geometry']['coordinates'])
                    p1, _ = nearest_points(pred_polygon, point)
                    d = point.distance(p1)
                    lng, lat = p1.x, p1.y

            # Print observation
            if mode == 'replay':
                print('>> %s was observed in %s on %s %s' %
                      (user, room, day, time))
            else:
                print('>> %s was just observed in %s' %
                      (user, room))
            print('Physical location:', (lat, lng))

            # Write results to history
            rel_hist.setdefault(user, []).append(loc)
            geo_hist.setdefault(user, []).append((lat, lng))
            sem_hist.setdefault(user, []).append(room)
            y_pred.append(pred)
            y_true.append(STATES.index(room))

            # Write to file if live
            if mode != 'replay':
                rss_data = json.dumps(rss_hist)
                loc_data = json.dumps(sem_hist)
                loc_f = open("locations.json", "w")
                rss_f = open("rss.json", "w")
                loc_f.write(loc_data)
                rss_f.write(rss_data)
                loc_f.close()
                rss_f.close()

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

    # Mode 1: Trilateration in real-time
    # while(True):
    #     run('live-all')

    # Fit HMM from JSON and make predications
    obs = json.loads(open('rss.json').read())
    labels = json.loads(open('locations.json').read())
    # m = classifier.train(obs, labels, 'rf')
    classifier.roc(obs, labels)

    # Mode 2: Replay historical data and parse observations to json
    # data = get_hist_data()
    # print('Data retrieved.')
    # global usernames
    # for r in data:
    #     if r['payload']['mac'] not in dict_of_macs:
    #         if usernames:
    #             random.shuffle(usernames)
    #             username = usernames.pop()
    #         else:
    #             username = 'user'+''.join(random.choices(string.digits, k=3))
    #         dict_of_macs[r['payload']['mac']] = username
    # window_end = convert_date_to_secs(TRILATERATION['end'])
    # for _ in range(window_start, window_end, TRILATERATION['window_size']):
    #     run('replay', data, m)
    # classifier.plot_confusion_matrix(y_true, y_pred)
    # classifier.plot_confusion_matrix(y_true, y_pred, normalize=True)

    # classifier.roc(y_true, y_pred)
    # plot_localization(sem_history)

    # Fit HMM from JSON and make predications
    # obs = json.loads(open('rss.json').read())
    # labels = json.loads(open('locations.json').read())
    # m = classifier.train(obs, labels, 'rf')
    # while(True):
    #     run('live', model=m)

    # while(True):
    #     run('mqtt', model=m)

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
        classifier.plot_confusion_matrix(y_true, y_pred)
        classifier.plot_confusion_matrix(y_true, y_pred, normalize=True)