import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime
from time import sleep
import json
import numpy as np
import math
import statistics
from config import TRILATERATION
from draw import draw
import scipy
from scipy.optimize import least_squares
from heapq import nlargest

dynamodb = boto3.resource('dynamodb')
tableIoT = dynamodb.Table('db_demo')


def estimate_distance_fspl(rss):
    """
    http://goo.gl/cGXmDw
    """
    rss = int(round(rss))
    logd = (27.55 - (20 * math.log10(2400)) + abs(rss)) / 20
    d = math.pow(10, logd)
    return d


def estimate_distance_itu(rss):
    """
    https://en.wikipedia.org/wiki/ITU_model_for_indoor_attenuation
    """
    rss = int(round(rss))
    f = 2400
    p_fn = 4
    N = 28
    logd = (abs(rss) - (20 * math.log10(f) + p_fn - 28)) / N
    d = math.pow(10, logd)
    return d


def estimate_distance_log(rss, gamma):
    """
    https://en.wikipedia.org/wiki/Log-distance_path_loss_model
    """
    rss = int(round(rss))
    pl0 = -29
    d0 = 1
    gamma = gamma
    Xg = np.random.standard_normal(1)[0]
    logdd0 = (abs(rss) - abs(pl0) - Xg) / (10 * gamma)
    dd0 = math.pow(10, logdd0)
    d = dd0 * d0
    return d


def distance(p1, p2):
    """
    Computes Pythagorean distance
    """
    d = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return d


def parameter_fitting(dict_of_rss):
    gammas = np.arange(2.0, 6.0, 0.1)
    for gamma in gammas:
        print("For gamma = ", gamma)
        dict_of_distances = {}
        for ap, rss in dict_of_rss.items():
            estimated_distance = estimate_distance_log(rss, gamma)
            dict_of_distances[ap] = estimated_distance
            print('The estimated distance of the AP %d is %f' %
                  (ap, estimated_distance))


def get_data_by_mac_address(mode, mac, APs):
    """
    Query data from AWS using mac address
    """
    dict_of_processed_rss = {}

    for ap in APs:

        # Query historical data in a specified time range
        if mode == "hist":
            start_timestamp = datetime.strptime(
                TRILATERATION['start_timestamp'], '%d %b %Y %H:%M')
            end_timestamp = datetime.strptime(
                TRILATERATION['end_timestamp'], '%d %b %Y %H:%M')

            start_in_sec = int(round(start_timestamp.timestamp()))
            end_in_sec = int(round(end_timestamp.timestamp()))

            response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap['id'])
                                      & Key('timestamp').between(start_in_sec, end_in_sec))

            # rss = compute_avg_rss_for_mac_address(response, mac)
            rss = compute_median_rss_for_mac_address(response, mac)
            dict_of_processed_rss[ap['id']] = rss

        # Query real-time data
        elif mode == "live":
            now_in_sec = int(round(datetime.now().timestamp()))
            response = tableIoT.query(KeyConditionExpression=Key('sensor_id').eq(ap['id'])
                                      & Key('timestamp').gte(now_in_sec-60))
            rss = get_live_rss_for_mac_address(response, mac)

            if rss == -1:
                print("warning: no live data detected for AP", ap['id'])
            else:
                dict_of_processed_rss[ap['id']] = rss

    return dict_of_processed_rss


def compute_avg_rss_for_mac_address(response, mac):
    sum_rss = 0
    count_rss = 0
    print(response)
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
        res += (distance((x, y), (xi, yi)) / abs(ri),)
    return res


def trilaterate_least_squares(guess):
    ls = least_squares(residuals, guess)
    return(ls.x)


def run(mode):
    # Query RSS data by mac address
    dict_of_rss = get_data_by_mac_address(
        mode, TRILATERATION['mac'], TRILATERATION['APs'])

    # Distance estimates dictionary
    global r
    r = {}
    for ap, rss in dict_of_rss.items():
        estimated_distance = estimate_distance_fspl(rss)
        # estimated_distance = estimate_distance_itu(rss)
        # estimated_distance = estimate_distance_log(rss, 2.3)
        r[ap] = estimated_distance
        print('The estimated distance of the AP %d is %f' %
              (ap, estimated_distance))

    # Points dictionary
    global p
    p = {}
    for i in r:
        p[i] = next(item for item in TRILATERATION['APs']
                    if item['id'] == i)['xy']

    # c = [29, 31, 34]
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
    draw(estimated_localization, localization, p, r)


def main():

    # Mode 1: Trilateration on historical data
    run("hist")

    # Mode 2: Trilateration in real-time
    while(True):
        run("live")
        sleep(1)


if __name__ == "__main__":
    main()
