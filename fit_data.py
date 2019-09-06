import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from config import CURVE, CURVES
from kalman_filter import KalmanFilter
from utils import convert_date_to_secs, get_rss_fluctuation


def func(rss, pl0, gamma):
    """
    Log distance function
    """
    logdd0 = (np.abs(rss) - abs(pl0)) / (10 * gamma)
    d = np.power(10, logdd0)
    return d


def fit():

    # Parse data from config file
    t, y, _ = get_rss_fluctuation(
        CURVE['0m_start'], CURVE['12m_end'], CURVE['ap'], CURVE['mac'])

    # Apply Kalman filter
    kalman = KalmanFilter(0.01, 0.1)
    filtered_y = []
    for p in y:
        filtered_y.append(kalman.filter(int(p)))

    # Plot raw data
    # plt.plot(filtered_y, range(0, len(y)))
    # plt.xlabel('RSS')
    # plt.ylabel('Sample')
    # plt.show()

    time_ranges = []
    for i in range(13):
        start_timestamp = convert_date_to_secs(CURVE[str(i)+'m_start'])
        end_timestamp = convert_date_to_secs(CURVE[str(i)+'m_end'])
        time_ranges.append((start_timestamp, end_timestamp))

    # Compute average RSS for every distance
    data = []
    avgs = []
    medians = []
    i = 0
    for s, e in time_ranges:
        single_data = []
        for p in t:
            if p in range(s, e):
                single_data.append(filtered_y[t.index(p)])
        data.append(single_data)
        medians.append(statistics.median(single_data))
        avgs.append(round(statistics.mean(single_data), 1))

        # Plot raw data histograms
        # plt.hist(single_data, bins=20, histtype='bar', label=i)
        i += 1

    plt.hist(data[:6], bins=30, label=range(0, 6))
    plt.xlabel('RSS')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('RSS frequencies measured in the first 6 meters')
    plt.show()

    # Plot collective RSS histogram
    plt.hist(y, bins=30)
    plt.xlabel('RSS')
    plt.ylabel('Frequency')
    plt.show()

    # Plot averaged data
    plt.plot(avgs, range(0, len(avgs)))

    # Fit curve
    popt, _ = curve_fit(func, avgs, range(0, len(avgs)))

    # Plot curve
    avgs.sort()
    print(avgs)
    plt.plot(avgs, func(avgs, *popt), 'g--',
             label='fit: RSS=%5.3f, gamma=%5.3f' % tuple(popt))
    plt.xlabel('RSS')
    plt.ylabel('Distance in meters')
    plt.legend()
    plt.show()


def fit_multiple():

    # Parse data from config file
    for mac in CURVES['macs']:
        for ap in CURVES['aps']:
            t, y, _ = get_rss_fluctuation(
                CURVES['1m_start'], CURVES['11m_end'], ap, mac)

            # Apply Kalman filter
            kalman = KalmanFilter(0.01, 0.1)
            filtered_y = []
            for p in y:
                filtered_y.append(kalman.filter(int(p)))

            # Plot raw data
            plt.plot(filtered_y, range(0, len(y)))
            plt.xlabel('RSS')
            plt.ylabel('Sample')
            plt.title('Raw data from ap %s for %s' % (ap, mac))
            # plt.show()

            time_ranges = []
            for i in range(1, 12, 2):
                start_timestamp = convert_date_to_secs(
                    CURVES[str(i)+'m_start'])
                end_timestamp = convert_date_to_secs(CURVES[str(i)+'m_end'])
                time_ranges.append((start_timestamp, end_timestamp))

            # Compute average RSS for every distance
            data = []
            avgs = []
            medians = []
            i = 0
            for s, e in time_ranges:
                single_data = []
                for p in t:
                    if p in range(s, e):
                        single_data.append(filtered_y[t.index(p)])
                data.append(single_data)
                # print(single_data)
                if single_data:
                    medians.append(statistics.median(single_data))
                    avgs.append(round(statistics.mean(single_data), 1))
                else:
                    medians.append(0)
                    avgs.append(0)

                # Plot raw data histograms
                # plt.hist(single_data, bins=20, histtype='bar', label=i)
                i += 1

            print(avgs)
            plt.hist(data, bins=30, label=range(1, 12, 2))
            plt.xlabel('RSS')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')
            plt.title('RSS frequencies from ap %s for %s' % (ap, mac))
            # plt.show()

            # Plot collective RSS histogram
            plt.hist(y, bins=30)
            plt.xlabel('RSS')
            plt.ylabel('Frequency')
            plt.title('Collective RSS from ap %s for %s' % (ap, mac))
            # plt.show()

            # Get only non-zero data
            avgs = [x for (i, x) in enumerate(avgs) if x != 0]
            distances = [i for (i, x) in enumerate(avgs) if x != 0]

            # Plot averaged data
            plt.plot(avgs, distances)

            # Fit curve
            popt, _ = curve_fit(func, avgs, distances)

            # Plot curve
            avgs.sort()
            # print('RSS averages for ap %s and device %s: %s' % (ap, mac, avgs))
            plt.plot(avgs, func(avgs, *popt), 'g--',
                     label='fit: RSS=%5.3f, gamma=%5.3f' % tuple(popt))
            plt.xlabel('RSS')
            plt.ylabel('Distance in meters')
            plt.title('Curve fit for ap %s and device %s' % (ap, mac))
            plt.ylim(0, 12)
            plt.legend()
            # plt.show()
