import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
import time
import json
import numpy as np
import math
import statistics
from config import TRILATERATION, RSS
from draw import draw
import scipy
from scipy.optimize import least_squares
from heapq import nlargest
from pathLoss import log
import matplotlib
import matplotlib.pyplot as plt
from kalmanFilter import KalmanFilter
from statistics import stdev


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

    # plt.hist(y, color = 'blue', edgecolor = 'black', bins=int(len(t)/10))
    # plt.title('Histogram of RSS at 1m')
    # plt.xlabel('RSS')
    # plt.show()

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


def get_closest_access_points():
    return sorted(r, key=r.get)[:3]


def distance(p1, p2):
    """
    Computes Pythagorean distance
    """
    d = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return d


def trilaterate(P1, P2, P3, r1, r2, r3):
    """
    https://bit.ly/2w3ybNU
    https://bit.ly/2EbDLSC
    """
    A = 2*P2[0] - 2*P1[0]  # 2(x2) - 2(x1)
    B = 2*P2[1] - 2*P1[1]  # 2(y2) - 2(y1)
    C = r1**2 - r2**2 - P1[0]**2 + P2[0]**2 - P1[1]**2 + P2[1]**2
    D = 2*P3[0] - 2*P2[0]
    E = 2*P3[1] - 2*P2[1]
    F = r2**2 - r3**2 - P2[0]**2 + P3[0]**2 - P2[1]**2 + P3[1]**2
    try:
        x = (C*E - F*B) / (E*A - B*D)
        y = (C*D - A*F) / (B*D - A*E)
    except ZeroDivisionError:
        print("error: division by zero, returning (0, 0)..")
        return (0, 0)
    return (x, y)


def residuals(guess):
    x, y = guess
    res = ()
    for i in p:
        xi = p[i][0]
        yi = p[i][1]
        ri = r[i]
        res += ((distance((x, y), (xi, yi)) - ri) / abs(ri), )
        # res += ((distance((x, y), (xi, yi))/ri - math.log(ri)), )
    return res


def trilaterate_least_squares(guess):
    ls = least_squares(residuals, guess)
    return(ls.x)


def run(mode):
    # Query RSS data by mac address
    dict_of_rss = get_data_by_mac_address(
        mode, TRILATERATION['mac'], TRILATERATION['aps'])

    # Distance estimates dictionary
    global r
    r = {}
    for ap, rss in dict_of_rss.items():
        estimated_distance = log(rss, 2.3)
        r[ap] = estimated_distance
        print('The estimated distance of the AP %d is %f' %
              (ap, estimated_distance))

    # Points dictionary
    global p
    p = {}
    for i in r:
        p[i] = next(item for item in TRILATERATION['aps']
                    if item['id'] == i)['xy']

    # c = [29, 31, 33]
    c = get_closest_access_points()
    print("Closest to access points", ', '.join(str(i) for i in c))

    # Trilateration
    p3 = {k: v for k, v in p.items() if k in c}
    r3 = {k: v for k, v in r.items() if k in c}
    if len(p3) == 3 and len(r3) == 3:
        args = (p3[c[0]], p3[c[1]], p3[c[2]], r3[c[0]], r3[c[1]], r3[c[2]])
        estimated_localization = trilaterate(*args)
    else:
        print("error: trilateration not possible")
        estimated_localization = (0, 0)
    print("Trilateration estimation: ", estimated_localization)
    localization = trilaterate_least_squares(estimated_localization)  # NLS
    print("NLS estimation: ", tuple(localization[:2]))

    # Draw
    if len(r3) >= 3:
        draw(estimated_localization, localization, p, r)


def main():

    # Mode 1: Trilateration on historical data
    # run("hist")

    # Mode 2: Trilateration in real-time
    # while(True):
    #     run("live")
    #     time.sleep(1)
    run_kalman_filter_rss()


if __name__ == "__main__":
    main()
