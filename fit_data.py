import statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from config import CURVE
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
        avgs.append(statistics.mean(single_data))

        # Plot raw data histograms
        # plt.hist(single_data, bins=20, histtype='bar', label=i)
        i += 1

    plt.hist(data[:6], bins=30, label=range(0, 6))
    plt.legend(loc='upper right')
    plt.show()

    # Plot collective RSS histogram
    plt.hist(y, bins=30)
    plt.xlabel('RSS')
    plt.show()

    # Plot averaged data
    plt.plot(avgs, range(0, len(avgs)))

    # Fit curve
    popt, _ = curve_fit(func, avgs, range(0, len(avgs)))

    # Plot curve
    avgs.sort()
    plt.plot(avgs, func(avgs, *popt), 'g--',
             label='fit: RSS=%5.3f, gamma=%5.3f' % tuple(popt))
    plt.xlabel('RSS')
    plt.ylabel('Distance in meters')
    plt.legend()
    plt.show()
