import itertools
import math
import statistics
from datetime import datetime

import boto3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyrebase
from boto3.dynamodb.conditions import Key
from shapely.geometry import Point, Polygon
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from config import FIREBASE, STATES, TRILATERATION
from mi import MAP

dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')
matplotlib.rcParams.update({'font.size': 20})


def distance(p1, p2):
    """
    Computes Pythagorean distance
    """
    d = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return d


def rotate(point, angle):
    """
    Rotate a point clockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    angle = math.radians(-angle)
    ox, oy = (0, 0)
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return (qx, qy)


def convert_date_to_secs(date):
    date = datetime.strptime(date, '%d %b %Y %H:%M')
    return int(date.timestamp())


def flatten(l):
    return list(itertools.chain(*l))


def compute_mean_rss(response, mac, ap):
    """
    Computes the mean RSS for a certain mac address over
      a time period if data exists otherwise returns -1
    """
    rss_values = []
    for r in response:
        if (r['payload']['mac']) == mac and r['payload']['sensor_id'] == ap:
            rss_values.append(r['payload']['rssi'])
    if rss_values:
        return statistics.mean(rss_values)
    return -1


def compute_median_rss(response, mac, ap):
    """
    Computes the median RSS for a certain mac address over
     a time period if data exists otherwise returns -1
    """
    rss_values = []
    for r in response:
        if (r['payload']['mac']) == mac and r['payload']['sensor_id'] == ap:
            rss_values.append(r['payload']['rssi'])
    if rss_values:
        return statistics.median(rss_values)
    return -1


def get_rss_fluctuation(start, end, ap, mac):
    start_in_sec = convert_date_to_secs(start)
    end_in_sec = convert_date_to_secs(end)

    response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap)
                              & Key('timestamp').between(start_in_sec, end_in_sec))

    avg_rss = compute_mean_rss(response['Items'], mac, ap)
    rss = []
    timestamps = []
    for r in response['Items']:
        if r['payload']['mac'] == mac and r['payload']['sensor_id'] == ap:
            rss.append(int(r['payload']['rssi']))
            timestamps.append(int(r['payload']['timestamp']))
    return(timestamps, rss, avg_rss)


def get_hist_data():
    data = []
    for ap in TRILATERATION['aps']:
        # Query historical data in a specified time range
        start_in_sec = convert_date_to_secs(TRILATERATION['start'])
        end_in_sec = convert_date_to_secs(TRILATERATION['end'])

        response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(
            ap['id']) & Key('timestamp').between(start_in_sec, end_in_sec))
        data.append(response['Items'])
    return flatten(data)


def get_live_data():
    data = []
    for ap in TRILATERATION['aps']:
        # Query real-time data
        now_in_sec = int(datetime.now().timestamp())
        response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(
            ap['id']) & Key('timestamp').gte(now_in_sec-TRILATERATION['window_size']))
        data.append(response['Items'])
    return flatten(data)


def get_live_rss_for_ap_and_mac_address(response, mac, ap):
    for r in response:
        if r['payload']['mac'] == mac and r['payload']['sensor_id'] == ap:
            return r['payload']['rssi']
    return -1


def mqtt_get_live_rss_for_ap_and_mac_address(response, mac, ap):
    for r in response:
        if r['mac'] == mac and r['sensor_id'] == ap:
            return r['rssi']
    return -1


def replay_hist_data(response, mac, ap, window_start):
    """
    Mimics live streaming for historical data
    """
    window_end = window_start + TRILATERATION['window_size']
    temp_dict = next((r for r in response if r['payload']['mac'] == mac and r['sensor_id'] ==
                      ap and r['payload']['timestamp'] >= window_start and r['payload']['timestamp'] <= window_end), -1)
    if temp_dict != -1:
        rss = temp_dict['payload']['rssi']
        del temp_dict
        return rss, response
    return -1, response


def read_from_firebase(tablename):
    firebase = pyrebase.initialize_app(FIREBASE)
    db = firebase.database()
    query = db.child(tablename).get()
    results = list(query.val().values())
    return results


def get_room_by_physical_location(lat, lng):
    """
    Returns the corresponding room name for a given physical location
    """
    point = Point(lng, lat)
    for room in MAP:
        polygon = Polygon(room['geometry']['coordinates'])
        if polygon.contains(point):
            return room['properties']['ref']
    return None


def plot_localization(history):
    """
    Plots a histogram with the occurences of users in rooms on a map
    and returns the results as a dictionary
    """
    rooms = list(set(flatten(history.values())))
    for k, v in history.items():
        counts = []
        for r in rooms:
            counts.append(list(v).count(r))
        plt.bar(rooms, counts, width=0.35, label=k)
    plt.xlabel('Room')
    plt.title('Localization in the time span from %s to %s' %
              (TRILATERATION['start'], TRILATERATION['end']))
    plt.legend(loc='upper right')
    plt.show()


def get_room_physical_location(room):
    room = next(d for (index, d) in enumerate(MAP)
                if d['properties']['ref'] == room)
    center = Polygon(room['geometry']['coordinates']).centroid
    return center.y, center.x


def get_closest_polygon(x, y):

    point = Point(x, y)
    min_dist = 10000
    closest_polygon = None
    closest_room = None

    for m in MAP:
        polygon = Polygon(m['geometry']['coordinates'])
        dist = polygon.distance(point)
        if dist < min_dist:
            min_dist = dist
            closest_polygon = polygon
            closest_room = m['properties']['ref']

    return closest_polygon, closest_room


def tidy_obs_data(X):
    data = []
    lengths = []
    for _, v in X.items():
        for e in v:
            # print(e, v)
            data.append([next((index for (index, d) in enumerate(MAP)
                               if d['properties']['ref'] == e))])
        lengths.append(len(v))
    data = np.ravel(data)
    return data, lengths


def tidy_rss(obs):
    data = []
    lengths = []
    last_obs = []
    for _, v in obs.items():
        single_user = []
        for e in v:
            single_obv = []
            for ap in TRILATERATION['aps']:
                e = {int(key): e[key] for key in e}
                if ap['id'] in e.keys():
                    single_obv.append(e[ap['id']])
                else:
                    if last_obs:
                        single_obv.append(last_obs[ap['id']])
                    else:
                        single_obv.append(-1)
            last_obs = single_obv
            single_user.append(single_obv)
        data.append(single_user)
        lengths.append(len(single_user))
    data = flatten(data)
    return data, lengths


def ml_plot(obs, pred, len_X, alg):
    X, len_X = tidy_obs_data(obs)
    obs_labels = [STATES[i] for i in X]
    pred_labels = [STATES[i] for i in pred]

    for index, l in enumerate(len_X):
        start_index = 0
        if index != 0:
            start_index = index+len_X[index-1]
        plt.plot(obs_labels[start_index:start_index+l], ".-", label="observations", ms=6,
                 mfc="blue", alpha=0.7)
        plt.legend(loc='best')
        plt.plot(pred_labels[start_index:start_index+l], ".-", label="predictions", ms=6,
                 mfc="orange", alpha=0.7)
        plt.legend(loc='best')
        plt.title('user=%s, alg=%s' % (list(obs.keys())[index], alg))
        plt.show()


def closest_access_points():
    p = (1.0, 7.0)
    aps_sorted = {}
    for ap in TRILATERATION['aps']:
        aps_sorted[ap['id']] = distance(p, ap['xy'])
    return sorted(aps_sorted.items(), key=lambda kv: kv[1])


def plot_confusion_matrix(y_true, y_pred, title=None, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    if not title:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    classes = np.array(STATES)[unique_labels(y_true, y_pred)]

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title=title,
           ylim=(len(classes)-0.5, -0.5))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
