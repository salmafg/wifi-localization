import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors.kde import KernelDensity

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
    """
    Plots from curve fitting experiment
    """

    # Parse data from config file
    t, X, _ = get_rss_fluctuation(
        CURVE['0m_start'], CURVE['12m_end'], CURVE['ap'], CURVE['mac'])

    # Apply Kalman filter
    kalman = KalmanFilter(0.01, 0.1)
    filtered_X = []
    for p in X:
        filtered_X.append(kalman.filter(int(p)))

    # Plot raw data
    # plt.figure()
    # plt.plot(filtered_X, range(0, len(y)))
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
                single_data.append(filtered_X[t.index(p)])
        data.append(single_data)
        medians.append(statistics.median(single_data))
        avgs.append(round(statistics.mean(single_data), 1))

        # Plot raw data histograms
        # plt.hist(single_data, bins=20, histtype='bar', label=i)
        i += 1

    plt.figure()
    plt.hist(data[:6], bins=30, label=range(0, 6))
    plt.xlabel('RSS')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('RSS frequencies measured in the first 6 meters')

    # Plot collective RSS histogram
    plt.figure()
    plt.hist(X, bins=30)
    plt.xlabel('RSS')
    plt.ylabel('Frequency')

    # Plot averaged data
    plt.figure()
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
    """
    Comprehensive plots from heterogentiy experiment
    """

    # Parse data from config file
    distances = range(1, 12, 2)
    for mac in CURVES['macs']:
        for ap in CURVES['aps']:
            t1, x1, _ = get_rss_fluctuation(
                CURVES['1m_start'], CURVES['5m_end'], ap, mac)
            t2, x2, _ = get_rss_fluctuation(
                CURVES['7m_start'], CURVES['11m_end'], ap, mac)
            t = t1 + t2
            X = x1 + x2

            # Apply Kalman filter
            kalman = KalmanFilter(0.01, 0.1)
            filtered_X = []
            for p in X:
                filtered_X.append(kalman.filter(int(p)))

            # Plot raw data
            plt.figure()
            plt.plot(X, range(0, len(X)))
            plt.xlabel('RSS')
            plt.ylabel('Sample')
            plt.title('Raw data from ap %s for %s' % (ap, mac))

            # Plot filtered data
            plt.figure()
            plt.plot(filtered_X, range(0, len(X)))
            plt.xlabel('RSS')
            plt.ylabel('Sample')
            plt.title('Filtered data from ap %s for %s' % (ap, mac))

            time_ranges = []
            for i in distances:
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
                        single_data.append(filtered_X[t.index(p)])
                data.append(single_data)
                medians.append(statistics.median(single_data))
                avgs.append(round(statistics.mean(single_data), 1))

                # Plot raw data histograms
                # plt.hist(single_data, bins=20, histtype='bar', label=i)
                i += 1

            # Color-coded histogram
            plt.figure()
            plt.hist(data, bins=30, label=distances)
            plt.xlabel('RSS')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')
            plt.title('RSS frequencies from ap %s for %s' % (ap, mac))

            # Plot collective RSS histogram
            plt.figure()
            plt.hist(filtered_X, bins=30, label=mac,
                     density=True, histtype='step')
            plt.xlabel('RSS')
            plt.ylabel('Frequency')
            plt.title('Collective RSS from ap %s for %s' % (ap, mac))

            # Kernel Density Estimation
            X = np.array(filtered_X)
            bandwidths = 10 ** np.linspace(-1, 1, 100)
            grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                {'bandwidth': bandwidths},
                                cv=LeaveOneOut.get_n_splits(X))
            grid.fit(X[:, None])
            bandwidth = grid.best_params_['bandwidth']
            X_d = np.linspace(-90, -10, X.shape[0])
            kde = KernelDensity(
                bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
            logprob = kde.score_samples(X_d[:, None])
            plt.fill_between(X_d, np.exp(logprob), alpha=0.2,
                             label='kernel', color='b')
            plt.xlabel('RSS')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.title('Kernel estimation for ap %d with bandwidth=%.2f' %
                      (ap, bandwidth))

            # Plot averaged data
            plt.figure()
            plt.plot(avgs, distances)

            # Fit curve
            popt, _ = curve_fit(func, avgs, distances)

            # Plot curve
            avgs.sort()
            print('RSS averages for ap %s and device %s: %s' % (ap, mac, avgs))
            plt.plot(avgs, func(avgs, *popt), 'g--',
                     label='fit: RSS=%5.3f, gamma=%5.3f' % tuple(popt))
            plt.xlabel('RSS')
            plt.ylabel('Distance in meters')
            plt.title('Curve fit for ap -%s and device %s' % (ap, mac))
            plt.ylim(0, 12)
            plt.legend()
            plt.show()


def fit_all():
    """
    Plots from heterogenity experiment
    """

    # Plotting the histograms of the different phones on every ap
    for i, ap in enumerate(CURVES['aps']):
        plt.figure()
        for mac in CURVES['macs']:
            _, x1, _ = get_rss_fluctuation(
                CURVES['1m_start'], CURVES['5m_end'], ap, mac)
            _, x2, _ = get_rss_fluctuation(
                CURVES['7m_start'], CURVES['11m_end'], ap, mac)
            X = x1 + x2

            # Apply Kalman filter
            kalman = KalmanFilter(0.01, 0.1)
            filtered_X = []
            for p in X:
                filtered_X.append(kalman.filter(int(p)))
            X = filtered_X

            # Data Histogram
            # plt.figure()
            # plt.hist(X, bins=30, label=mac, histtype='step', density=True)
            # plt.xlabel('RSS')
            # plt.ylabel('Frequency')
            # plt.legend()
            # plt.title('Collective RSS for ap %d' % ap)

            # Kernel Density Estimation
            X = np.array(X)
            # bandwidths = 10 ** np.linspace(-1, 1, 100)
            # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
            #                     {'bandwidth': bandwidths},
            #                     cv=LeaveOneOut().get_n_splits(X))
            # grid.fit(X[:, None])
            # bandwidth = grid.best_params_['bandwidth']
            # print(grid.best_params_)
            bandwidth = 2.0
            X_d = np.linspace(-90, -10, X.shape[0])
            kde = KernelDensity(
                bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
            logprob = kde.score_samples(X_d[:, None])
            plt.fill_between(X_d, np.exp(logprob), alpha=0.2,
                             label=mac)
            plt.xlabel('RSS')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.title('Kernel estimation for ap %d with bandwidth=%.2f' %
                      (ap, bandwidth))
    plt.show()

    # Plotting the histograms of the different aps on every phone
    for i, mac in enumerate(CURVES['macs']):
        plt.figure()
        for ap in CURVES['aps']:
            _, x1, _ = get_rss_fluctuation(
                CURVES['1m_start'], CURVES['5m_end'], ap, mac)
            _, x2, _ = get_rss_fluctuation(
                CURVES['7m_start'], CURVES['11m_end'], ap, mac)
            X = x1 + x2

            # Apply Kalman filter
            kalman = KalmanFilter(0.01, 0.1)
            filtered_X = []
            for p in X:
                filtered_X.append(kalman.filter(int(p)))
            X = filtered_X

            # Data Histogram
            # plt.hist(X, bins=30, label=ap, histtype='step', stacked=True)
            # plt.xlabel('RSS')
            # plt.ylabel('Frequency')
            # plt.legend()
            # plt.title('Collective RSS for phone %s' % mac)

            # Kernel Density Estimation
            X = np.array(X)
            # bandwidths = 10 ** np.linspace(-1, 1, 100)
            # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
            #                     {'bandwidth': bandwidths},
            #                     cv=LeaveOneOut().get_n_splits(X))
            # grid.fit(X[:, None])
            # bandwidth = grid.best_params_['bandwidth']
            # print(grid.best_params_)
            bandwidth = 2.0
            X_d = np.linspace(-90, -10, X.shape[0])
            kde = KernelDensity(
                bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
            logprob = kde.score_samples(X_d[:, None])
            plt.fill_between(X_d, np.exp(logprob), alpha=0.2,
                             label=ap)
            plt.xlabel('RSS')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.title('Kernel estimation for device %s with bandwidth=%.2f' %
                      (mac, bandwidth))
    plt.show()
