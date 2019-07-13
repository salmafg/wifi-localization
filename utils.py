import itertools
import math
import statistics
from datetime import datetime

import boto3
import numpy as np
import pyrebase
from boto3.dynamodb.conditions import Key

from config import FIREBASE, TRILATERATION

dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')


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


def compute_mean_rss_for_mac_address(response, mac):
    """
    Computes the mean RSS for a certain mac address over
      a time period if data exists otherwise returns -1
    """
    rss_values = []
    for r in response:
        if (r['payload']['mac']) == mac:
            rss_values.append(r['payload']['rssi'])
    if rss_values:
        return statistics.mean(rss_values)
    return -1


def compute_median_rss_for_mac_address(response, mac):
    """
    Computes the median RSS for a certain mac address over
     a time period if data exists otherwise returns -1
    """
    rss_values = []
    for r in response:
        if (r['payload']['mac']) == mac:
            rss_values.append(r['payload']['rssi'])
    if rss_values:
        return statistics.median(rss_values)
    return -1


def get_rss_fluctuation_by_mac_address(start, end, ap):
    start_in_sec = convert_date_to_secs(start)
    end_in_sec = convert_date_to_secs(end)

    response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap)
                              & Key('timestamp').between(start_in_sec, end_in_sec))

    avg_rss = compute_mean_rss_for_mac_address(response, TRILATERATION['mac'])
    rss = []
    timestamps = []
    for r in response:
        if (r['payload']['mac']) == TRILATERATION['mac']:
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


def get_data_by_mac_address(mode, mac, APs):
    """
    Query data from AWS using mac address
    """
    dict_of_processed_rss = {}

    for ap in APs:

        # Query historical data in a specified time range
        if mode == "hist":
            start_in_sec = convert_date_to_secs(TRILATERATION['start'])
            end_in_sec = convert_date_to_secs(TRILATERATION['end'])

            response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap['id'])
                                      & Key('timestamp').between(start_in_sec, end_in_sec))

            # rss = compute_avg_rss_for_mac_address(response, mac)
            rss = compute_median_rss_for_mac_address(
                response['Items'], mac)
            dict_of_processed_rss[ap['id']] = rss

        # Query real-time data
        elif mode == "live":
            now_in_sec = int(datetime.now().timestamp())
            response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(
                ap['id']) & Key('timestamp').gte(now_in_sec-TRILATERATION['window_size']))
            rss = get_live_rss_for_mac_address(response['Items'], mac)

            if rss == -1:
                print("warning: no live data detected for AP", ap['id'])
            else:
                dict_of_processed_rss[ap['id']] = rss

    return dict_of_processed_rss


def get_live_rss_for_mac_address(response, mac):
    for r in response:
        if r['payload']['mac'] == mac:
            return r['payload']['rssi']
    return -1


def get_live_rss_for_ap_and_mac_address(response, mac, ap):
    for r in response:
        if r['payload']['mac'] == mac and r['payload']['sensor_id'] == ap:
            return r['payload']['rssi']
    return -1


def replay_hist_data(response, mac, ap, window_start):
    """
    Mimics live streaming for historical data
    """
    window_end = window_start + TRILATERATION['window_size']
    for r in response:
        r = r['payload']
        if r['mac'] == mac and r['sensor_id'] == ap and r['timestamp'] >= window_start and r['timestamp'] <= window_end:
            return r['rssi']
    return -1


def slope(p1, p2):
    """
    Computes the slope of two points
    """
    m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    return m


def angle(m1, m2):
    """
    Computes the angle between two lines in degrees
    """
    a = math.degrees(math.atan((m2 - m1) / (1 + m1 * m2)))
    return a


def read_from_firebase(tablename):
    firebase = pyrebase.initialize_app(FIREBASE)
    db = firebase.database()
    query = db.child(tablename).get()
    results = list(query.val().values())
    return results
