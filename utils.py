import math
from datetime import datetime
from config import TRILATERATION
import statistics
import boto3
from boto3.dynamodb.conditions import Key


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


def compute_mean_rss_for_mac_address(response, mac):
    rss_values = []
    for r in response['Items']:
        if (r['payload']['mac']) == mac:
            # print(r['payload'])
            rss_values.append(r['payload']['rssi'])
    avg_rss = statistics.mean(rss_values)
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


def get_rss_fluctuation_by_mac_address(start, end, ap):
    start_in_sec = convert_date_to_secs(start)
    end_in_sec = convert_date_to_secs(end)

    response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap)
                              & Key('timestamp').between(start_in_sec, end_in_sec))

    avg_rss = compute_mean_rss_for_mac_address(response, TRILATERATION['mac'])
    rss = []
    timestamps = []
    for r in response['Items']:
        if (r['payload']['mac']) == TRILATERATION['mac']:
            rss.append(int(r['payload']['rssi']))
            timestamps.append(int(r['payload']['timestamp']))
    return(timestamps, rss, avg_rss)


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
            rss = compute_median_rss_for_mac_address(response, mac)
            dict_of_processed_rss[ap['id']] = rss

        # Query real-time data
        elif mode == "live":
            now_in_sec = int(datetime.now().timestamp())
            response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(
                ap['id']) & Key('timestamp').gte(now_in_sec-TRILATERATION['time_window']))
            rss = get_live_rss_for_mac_address(response, mac)

            if rss == -1:
                print("warning: no live data detected for AP", ap['id'])
            else:
                dict_of_processed_rss[ap['id']] = rss

    return dict_of_processed_rss


def get_live_rss_for_mac_address(response, mac):
    for r in response['Items']:
        if (r['payload']['mac']) == mac:
            return r['payload']['rssi']
    return -1
