import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime
import json
import numpy as np
import math
import statistics
from config import TRILATERATION
from draw import draw_trilateration

dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')

# http://goo.gl/cGXmDw
def estimate_distance(rss):
    rss = int(round(rss))
    result = (27.55 - (20 * math.log10(2400)) + abs(rss)) / 20
    d = math.pow(10, result)
    return d

# Query data from AWS using mac address
def get_data_by_mac_address(mac, APs):

    dict_of_processed_rss = {}

    for ap in APs:

        start_timestamp = datetime.strptime(
            TRILATERATION['start_timestamp'], '%d %b %Y %H:%M')
        end_timestamp = datetime.strptime(
            TRILATERATION['end_timestamp'], '%d %b %Y %H:%M')

        start_in_sec = int(round(start_timestamp.timestamp()))
        end_in_sec = int(round(end_timestamp.timestamp()))

        # now_in_sec = int(round(datetime.now().timestamp()))

        response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap['id'])
            & Key('timestamp').between(start_in_sec, end_in_sec))

        # avg_rss = compute_avg_rss_for_mac_address(response, mac)
        # dict_of_processed_rss[ap['id']] = avg_rss

        median_rss = compute_median_rss_for_mac_address(response, mac)
        dict_of_processed_rss[ap['id']] = median_rss

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
    print(avg_rss)
    return avg_rss

def compute_median_rss_for_mac_address(response, mac):

    rss_values = []
    for r in response['Items']:
        if (r['payload']['mac']) == mac:
            # print(r['payload'])
            rss_values.append(r['payload']['rssi'])
    median_rss = statistics.median(rss_values)
    print(median_rss)
    return median_rss

# https://bit.ly/2w3ybNU
# https://bit.ly/2EbDLSC
def trilaterate(P1, P2, P3, r1, r2, r3):

    # p1 = np.array([0, 0, 0])
    # p2 = np.array([P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]])
    # p3 = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])
    # v1 = p2 - p1
    # v2 = p3 - p1

    # Xn = (v1)/np.linalg.norm(v1)

    # tmp = np.cross(v1, v2)

    # Zn = (tmp)/np.linalg.norm(tmp)

    # Yn = np.cross(Xn, Zn)

    # i = np.dot(Xn, v2)
    # d = np.dot(Xn, v1)
    # j = np.dot(Yn, v2)

    # X = ((r1**2)-(r2**2)+(d**2))/(2*d)
    # Y = (((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(X))
    # Z1 = np.sqrt(max(0, r1**2-X**2-Y**2))
    # Z2 = -Z1

    # K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    # K2 = p1 + X * Xn + Y * Yn - Z2 * Zn
    # return K1
    A = 2*P2[0] - 2*P1[0]
    B = 2*P2[1] - 2*P1[1]
    C = r1**2 - r2**2 - P1[0]**2 + P2[0]**2 - P1[1]**2 + P2[1]**2
    D = 2*P3[0] - 2*P2[0]
    E = 2*P3[1] - 2*P2[1]
    F = r2**2 - r3**2 - P2[0]**2 + P3[0]**2 - P2[1]**2 + P3[1]**2
    x = (C*E - F*B) / (E*A - B*D)
    y = (C*D - A*F) / (B*D - A*E)
    return (x, y)


dict_of_avgs_rss = get_data_by_mac_address(
    TRILATERATION['mac'], TRILATERATION['APs'])
dict_of_estimated_distances = {}
for ap, rss in dict_of_avgs_rss.items():
    estimated_distance = estimate_distance(rss)
    dict_of_estimated_distances[ap] = estimate_distance(rss)
    print('The estimated distance of the AP %d is %f' %
          (ap, estimated_distance))
P1 = TRILATERATION['APs'][0]['xy']
P2 = TRILATERATION['APs'][1]['xy']
P3 = TRILATERATION['APs'][2]['xy']
r1 = dict_of_estimated_distances[TRILATERATION['APs'][0]['id']]
r2 = dict_of_estimated_distances[TRILATERATION['APs'][1]['id']]
r3 = dict_of_estimated_distances[TRILATERATION['APs'][2]['id']]
localization = trilaterate(P1, P2, P3, r1, r2, r3)
print(localization)
draw_trilateration(P1[0], P1[1], r1, P2[0], P2[1], r2, P3[0], P3[1], r3, localization)