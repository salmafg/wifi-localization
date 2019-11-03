import csv
import json
import pickle
import random
import string
import subprocess

import numpy as np
import pyrebase
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

import classifier
import eval
import gdop
import hmm
import kmeans
from config import *
from draw import draw
from fit_data import *
from kalman_filter import KalmanFilter
from mqtt import subscriber
from nls import nls
from path_loss import log
from trilateration import trilaterate
from utils import *

firebase = pyrebase.initialize_app(FIREBASE)
db = firebase.database()

dict_of_macs = TRILATERATION['macs']
window_start = convert_date_to_secs(TRILATERATION['start'])
rel_hist = {}
try:
    sem_hist = json.loads(open('data/hist/semantic3.json').read())
except:
    sem_hist = {}
last_rss = [-60]*len(TRILATERATION['aps'])
y_true = []
y_pred = []


with open('data/usernames.csv', 'r') as f:
    reader = csv.reader(f)
    usernames = flatten(list(reader))


def run(mode, data=None, model=None, record=False, broadcast=False, polygons=False, project=True, evaluate=[]):
    '''
    Runs localization for multiple mac devices
    '''
    global usernames
    global window_start
    dict_of_mac_rss = {}
    timestamp = datetime.now()

    if mode == 'live':
        data = get_live_data()
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = get_live_rss_for_ap_and_mac_address(data, mac, ap['id'])
                dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss

    elif mode == 'mqtt':
        data = subscriber.get_messages()
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss = mqtt_get_live_rss_for_ap_and_mac_address(
                    data, mac, ap['id'])
                dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss

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
                dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss

    elif mode == 'mqtt-all':
        data = subscriber.get_messages()
        for r in data:
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
                dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss

    elif mode == 'replay':
        timestamp = datetime.fromtimestamp(
            window_start)
        for mac, user in dict_of_macs.items():
            dict_of_rss = {}
            for ap in TRILATERATION['aps']:
                rss, data = replay_hist_data(data, mac, ap['id'], window_start)
                dict_of_rss[ap['id']] = round(rss)
            if dict_of_rss:
                dict_of_mac_rss[mac] = dict_of_rss
        window_start += TRILATERATION['stride_size']

    else:
        raise ValueError('invalid run mode `%s` ' % mode)

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

                # print(evaluate) # distance estimation
                # with open(evaluate[0], 'a') as csv_file:
                #     writer = csv.writer(csv_file)
                #     writer.writerow([user, ap, rss, evaluate[1], round(estimated_distance, 3)])

        # Points dictionary
        p = {}
        for i in r:
            p[i] = next(item for item in TRILATERATION['aps']
                        if item['id'] == i)['xy']

        c = sorted(r, key=r.get)[:3]
        if c:
            print('Closest to access points:', ', '.join(str(i) for i in c))

        p3 = {k: v for k, v in p.items() if k in c}
        r3 = {k: v for k, v in r.items() if k in c}
        localization = None

        if len(p3) == 3:

            # Trilateration
            # args = (p3[c[0]], p3[c[1]], p3[c[2]], r3[c[0]], r3[c[1]], r3[c[2]])
            # estimated_localization = trilaterate(*args)
            estimated_localization = (0, 0)
            print('Trilateration estimate:', estimated_localization)

            # Non-linear least squares
            localization = nls(estimated_localization, p, r)
            print('NLS estimate:', tuple(localization[:2]))

            # Compute uncertainty
            uncertainty = min(r.values())
            if user in rel_hist.keys():
                delta_t_1 = timestamp - \
                    rel_hist[user][len(rel_hist[user])-1][1]
                d_1 = distance(
                    localization, rel_hist[user][len(rel_hist[user])-1][0])
                v_1 = d_1 / delta_t_1.total_seconds()
                print('v_1:', v_1)
                if v_1 <= TRILATERATION['velocity_threshold']:
                    # Distance between current and previous localization is small enough
                    uncertainty = max(uncertainty, d_1)

                # If distance was too large and deeper history exists
                elif len(rel_hist[user]) >= 2:
                    delta_t_2 = timestamp - \
                        rel_hist[user][len(rel_hist[user])-2][1]
                    d_2 = distance(
                        localization, rel_hist[user][len(rel_hist[user])-2][0])
                    v_2 = d_2 / delta_t_2.total_seconds()
                    print('v_2:', v_2)
                    if v_2 <= TRILATERATION['velocity_threshold']:
                        # Distance from deeper history was small enough (_1 was an anomaly)
                        uncertainty = max(uncertainty, d_2)
                    else:
                        # Take max of current and previous because previous is more relevant than _2
                        uncertainty = max(uncertainty, d_1)
            if uncertainty < TRILATERATION['minimum_uncertainty']:
                uncertainty = TRILATERATION['minimum_uncertainty']
            uncertainty = round(uncertainty, 3)
            print('Uncertainty: %.1fm' % round(uncertainty, 1))

            # Draw
            # draw(estimated_localization, localization, p, r)

            # Correct angle deviation
            corrected_localization = rotate(localization, GEO['deviation'])

            # Compute geographic location
            lat = GEO['origin'][0] + \
                corrected_localization[1]*GEO['oneMeterLat']
            lng = GEO['origin'][1] + \
                corrected_localization[0]*GEO['oneMeterLng']

            # Move invalid point inside building to a valid location
            room = get_room_by_physical_location(lat, lng)
            if polygons and room is None:
                closest_polygon, closest_room = get_closest_polygon(lng, lat)
                p1, _ = nearest_points(closest_polygon, Point(lng, lat))
                p1_rel_x = (p1.x - GEO['origin'][1]) / GEO['oneMeterLng']
                p1_rel_y = (p1.y - GEO['origin'][0]) / GEO['oneMeterLat']
                corrected_rel_loc = rotate(
                    (p1_rel_x, p1_rel_y), -GEO['deviation'])
                d = (Point(corrected_rel_loc)).distance(Point(localization))
                lng, lat = p1.x, p1.y
                room = closest_room
                print('...point was moved %.3fm' % d)

            if evaluate:
                print(evaluate)
                with open(evaluate[0], 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    # writer.writerow([user] + evaluate[2:len(evaluate)] +
                    #                 list(localization) + list(corrected_rel_loc) +
                    #                 [evaluate[1], room, closest_room, uncertainty])
                    writer.writerow([user]+evaluate[1:3] +
                                    [len([i for i in dict_of_rss.values() if i > TRILATERATION['rss_threshold']])] +
                                    evaluate[3:]+list(localization) + list(corrected_rel_loc) + [uncertainty])  # thres
                    # writer.writerow([user] + evaluate[1:]+
                    # list(localization) + list(corrected_rel_loc)+ [uncertainty])  # pof
                    # truth_local = next(
                    #     item for item in data if item['timestamp'] >= timestamp.timestamp() and item['timestamp'] <= timestamp.timestamp()+TRILATERATION['window_size'])
                    # rel_x = (truth_local['longitude'] - GEO['origin'][1]) / GEO['oneMeterLng']
                    # rel_y = (truth_local['latitude'] - GEO['origin'][0]) / GEO['oneMeterLat']
                    # writer.writerow([rel_x, rel_y]+list(localization)+[distance((rel_x, rel_y), localization)])
                csv_file.close()

            # Machine learning prediction
            if model is not None:
                temp = list(dict_of_rss.values())
                temp = np.atleast_2d(temp)
                pred, prob = classifier.predict_room(model, temp)
                print('>> model prediction in %s with probability %f' %
                      (STATES[pred], prob))
                if prob >= ML['prob_threshold'] and room != pred:
                    point = Point(lng, lat)
                    pred_polygon = Polygon(
                        MAP[pred]['geometry']['coordinates'])
                    p1, _ = nearest_points(pred_polygon, point)
                    d = point.distance(p1)
                    lng, lat = p1.x, p1.y
                y_pred.append(pred)
                y_true.append(STATES.index(room))

            # Print observation
            user = list(dict_of_macs.values())[
                list(dict_of_macs.keys()).index(mac)]
            if mode == 'replay':
                print('>> %s was observed in %s on %s' %
                      (user, room, timestamp.strftime('%d %b %H:%M:%S')))
            else:
                print('>> %s was just observed in %s' %
                      (user, room))
            print('Physical location:', (lat, lng))

            # Write results to history
            rel_hist.setdefault(user, []).append((localization, timestamp))
            sem_hist.setdefault(user, []).append(room)

            # Write to file
            if record:
                row = list(dict_of_rss.values())
                for i, _ in enumerate(row):
                    if row[i] == -1:
                        row[i] = last_rss[i]
                    else:
                        last_rss[i] = row[i]
                row.insert(0, room)
                with open(ML['data'], 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)
                csv_file.close()

            # Push data to Firebase
            if project:
                data = {
                    'mac': mac,
                    'user': user,
                    'lat': lat,
                    'lng': lng,
                    'radius': str(uncertainty),
                    'timestamp': timestamp.strftime('%d %b %H:%M:%S')
                }
                db.child(FIREBASE['table']).child(mac).set(data)
                # db.child(FIREBASE['table']).push(data)

            if broadcast:
                msg = {
                    'mac': mac,
                    'latitude': lat,
                    'longitude': lng,
                    'timestamp': timestamp,
                    'floor': 0,
                    'radius': uncertainty
                }
                publisher.send_message(msg)

        elif localization is not None:
            print('info: trilateration not possible, using last value', localization)

        if model is not None:
            for k, v in dict_of_rss.items():
                if v != -1:
                    last_rss[k] = v

        # HMM
        # data = json.dumps(sem_hist)
        # f = open('data/hist/semantic3.json', "w")
        # f.write(data)
        # f.close()


def main():

    # # Train classifier and make predications
    # m = classifier.train('knn')
    train_x = json.loads(open('data/hist/train_x.json').read())
    train_y = json.loads(open('data/hist/train_y.json').read())
    test_x = json.loads(open('data/hist/semantic1.json').read())
    test_y = json.loads(open('data/hist/truth1.json').read())
    # m = hmm.create(train_x) # create or fit then predict
    # hmm.predict_all(m, test_x, test_y, 'map')
    hmm.train(train_x, train_y, test_x, test_y,
              training='labeled', decoder='viterbi')

    # Mode 1: Trilateration in real-time
    # while True:
    #     run(mode='mqtt-all', record=False, polygons=True, project=False)

    # Mode 2: Replay historical data and parse observations to json
    # t = closest_access_points((1, 7.0))
    # print(t)
    # x = 28 - len(TRILATERATION['aps'])
    # print('x =', len(TRILATERATION['aps']))
    # eval.point_of_failure("data/eval/pof/pof_1.csv")
    # print(centroid())
    # plt.figure()
    # eval.point_of_failure("data/eval/pof/pof_2.csv")
    # plt.show()
    # eval.uncertainty("data/eval/threshold/full.csv")

    # eval.localization_error("data/eval/localization/trilat_only.csv")
    # eval.localization_error("data/eval/localization/nls_trilat.csv")
    # eval.localization_error("data/eval/localization/wnls_dist_trilat.csv")
    # eval.localization_error("data/eval/localization/wnls_var_trilat.csv")
    # eval.localization_error("data/eval/localization/nls_init_0.csv")
    # eval.localization_error("data/eval/localization/wnls_dist_0.csv")
    # eval.localization_error("data/eval/localization/wnls_var_0.csv")
    # eval.localization_error("data/eval/threshold/full.csv")

    # eval.rssi_threshold("data/eval/threshold/full.csv")
    # eval.localization_error('dubai_local_70.csv')
    # eval.distance_estimation('data/eval/distance/dlos/curve2.csv', 'data/eval/distance/nlos/curve2.csv')

    # data = get_hist_data()
    # print('Data retrieved.')
    # with open('outfile', 'wb') as f:
    #     pickle.dump(data, f)

    # with open('outfile', 'rb') as f:
    #     data = pickle.load(f)
    # # global usernames
    # # for r in data:
    # #     if r['payload']['mac'] not in dict_of_macs:
    # #         if usernames:
    # #             random.shuffle(usernames)
    # #             username = usernames.pop()
    # #         else:
    # #             username = 'user'+''.join(random.choices(string.digits, k=3))
    # #         dict_of_macs[r['payload']['mac']] = username
    # window_end = convert_date_to_secs(TRILATERATION['end'])
    # for _ in range(window_start, window_end, TRILATERATION['window_size']):
    #     run('replay', data, project=True, polygons=False)
    #     evaluate=['data/eval/distance/nlos/curve2.csv', 11])
    #     evaluate=['data/eval/uncertainty_velocity.csv', 0.875, 15.0])
    #         #  [-85, 80, -75, -70, -65, -60, -55]
    # evaluate=['data/eval/threshold/full.csv', TRILATERATION['rss_threshold'],
    #  9, -2.4, 1.76])
    #         evaluate=['data/eval/pof/pof_2_corrected.csv', 3, round(t[0][1], 2), 0.875, 15.0])
    # evaluate=['data/eval/wnls_var_0.csv', 'the corridor', 1, 1.0, 7.0])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.053', 2, -2.0, 5.1])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.056', 3, 3.6, 9.5])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.065', 4, -2.6, 27.0])
    #     # evaluate=['data/eval/wnls_var_0.csv', 'the corridor', 5, 0.5, 22.5])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.055', 6, -2.8, 9.55])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.059', 7, -2.0, 17.0])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.062', 8, 3.7, 19.2])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.051', 9, -2.4, 1.76])
    #     # evaluate=['data/eval/wnls_var_0.csv', 'the corridor', 10, 0.0, 1.8])
    #     # evaluate=['data/eval/wnls_var_0.csv', '00.11.056', 3, 2.342, 8.97])
    # print(closest_access_points((0.875, 15.0)))
    # plot_localization(sem_hist)

    # Fit curve
    # fit()
    # fit_multiple()
    # fit_all()
    # plot_rssi_dist()
    # heterogeneity_scatter()
    # eval.plot_localization_error()

    # Kalman filter
    # run_kalman_filter_rss()

    # kmeans.cluster()

    # print(classifier.train('knn'))
    # classifier.roc()


if __name__ == '__main__':
    try:
        main()
        subprocess.Popen(['notify-send', "Localization complete."])
        if y_true:
            plot_confusion_matrix([5]*len(y_pred), y_pred)
    except KeyboardInterrupt:
        if y_true:
            plot_confusion_matrix(y_true, y_pred)
        else:
            pass
