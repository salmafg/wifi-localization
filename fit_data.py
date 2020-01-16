from itertools import combinations
from statistics import mean, median

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy
from scipy.optimize import curve_fit
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors.kde import KernelDensity

from config import CURVE, CURVES, TRILATERATION
from kalman_filter import KalmanFilter
from utils import convert_date_to_secs, get_rss_fluctuation

matplotlib.rcParams.update({
    'font.size': 22,
    'font.family': 'serif',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'legend.fontsize': 'small',
    'figure.autolayout': True,
    'figure.figsize': (12, 8)
})


def log_dist_func(rss, pl0, gamma):
    """
    Log distance function
    """
    logdd0 = (np.abs(rss) - abs(pl0)) / (10 * gamma)
    d = np.power(10, logdd0)
    return d

def log_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def lin_func(x, b, c):
    """
    Linear function
    """
    return np.multiply(b, x) + c


def quad_func(x, a, b, c):
    """
    Quadratic function
    """
    return np.multiply(a, np.power(x, 2)) + np.multiply(b, x) + c


def poly_func(x, a, b, c, d):
    """
    Polynomial function
    """
    return np.multiply(d, np.power(x, 3)) + np.multiply(a, np.power(x, 2)) + np.multiply(b, x) + c


def plot_rssi_dist():
    X = []
    for ap in range(0, 30):
        _, x1, _ = get_rss_fluctuation(
            "4 Oct 2019 16:32", "4 Oct 2019 16:55", ap, "")
        X += x1
    plt.figure()
    # plt.hist(X, bins=30, density=True, histtype='step')
    bandwidth = 2
    X = np.array(X)
    X_d = np.linspace(-100, 0, X.shape[0])
    kde = KernelDensity(
        bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
    logprob = kde.score_samples(X_d[:, None])
    plt.fill_between(X_d, np.exp(logprob), alpha=0.2, color='b')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Probability Density')
    plt.xlim(-100, 0)
    plt.show()


def fit():
    """
    Plots from curve fitting experiment
    """

    # Parse data from config file
    t = []
    X = []
    for i in range(0, 13):
        t_temp, X_temp, _ = get_rss_fluctuation(
            CURVE[str(i)+'m_start'], CURVE[str(i)+'m_end'], CURVE['ap'], CURVE['mac'])
        t = t + t_temp
        X = X + X_temp
    print(len(t), len(X))

    # Apply Kalman filter
    kalman = KalmanFilter(0.01, 0.1)
    filtered_X = []
    for p in X:
        filtered_X.append(kalman.filter(int(p)))

    time_ranges = []
    for i in range(13):
        start_timestamp = convert_date_to_secs(CURVE[str(i)+'m_start'])
        end_timestamp = convert_date_to_secs(CURVE[str(i)+'m_end'])
        time_ranges.append((start_timestamp, end_timestamp))

    # Compute average RSSI for every distance
    data = []
    hist_data = []
    avgs = []
    medians = []
    i = 0
    for s, e in time_ranges:
        single_data = []
        for p in t:
            if p in range(s, e):
                single_data.append(X[t.index(p)])
        data = data + list(zip(single_data, [i]*len(single_data)))
        hist_data.append(single_data)
        i += 1
        medians.append(median(single_data))
        avgs.append(round(mean(single_data), 1))

    print(len(X))
    print(len(data))
    lengths = [len(x) for x in hist_data]
    # print(lengths)
    print(sum(lengths))
    ticks = [0]
    for i, l in enumerate(lengths):
        # print(i, l)
        ticks.append(sum(lengths[:i+1]))
    # del ticks[-1]
    print(ticks)

    # Plot raw data
    plt.xticks(ticks, range(0, 14))
    plt.plot(range(0, len(X)), X)
    plt.ylabel('RSSI (dBm)')
    plt.xlabel('Distance (m)')

    plt.figure()
    plt.hist(hist_data[:6], bins=30, label=range(0, 6), density=True)
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Probability Density')
    plt.legend()
    # plt.title('RSSI frequencies measured in the first 6 meters')

    plt.figure()
    plt.hist(X, bins=20,
                density=True, histtype='step')
    plt.xlabel('RSSI (dB)')
    plt.ylabel('Frequency')
    bandwidth = 2
    X = np.array(X)
    X_d = np.linspace(-75, -25, X.shape[0])
    kde = KernelDensity(
        bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
    logprob = kde.score_samples(X_d[:, None])
    plt.fill_between(X_d, np.exp(logprob), alpha=0.2, color='b')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Probability Density')
    # plt.legend(loc='upper right')

    # Plot collective RSSI histogram
    # plt.figure()
    # plt.hist(X, bins=30)
    # plt.xlabel('RSSI')
    # plt.ylabel('Frequency')

    # Plot averaged data
    # plt.figure()
    # plt.plot(avgs, range(0, len(avgs)))

    plt.figure()
    x, y = zip(*sorted(data, key=lambda x: x[0]))
    plt.scatter(y, x)
    popt, _ = curve_fit(log_dist_func, x, y)

    # Fit curve
    x, y = zip(*sorted(zip(avgs, range(0, len(avgs)))))
    popt, _ = curve_fit(log_dist_func, x, y)

    # Plot curve
    print('Collective RSSI averages: %s' % avgs)
    x_fit = np.linspace(0, -100, 100)
    plt.plot(log_dist_func(x_fit, *popt), x_fit, 'g--',
             label='fit: $PL_0$=-%5.3f, $\gamma$=%5.3f' % tuple(popt))
    plt.ylabel('RSSI (dBm)')
    plt.xlabel('Distance (m)')
    plt.xlim(-1, 25)
    plt.ylim(-80, 0)
    plt.legend(loc='upper right')
    plt.show()


def fit_multiple():
    """
    Comprehensive plots from heterogentiy experiment
    """

    # Parse data from config file
    distances = range(1, 12, 2)
    all_avgs = []
    all_medians = []
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
            # plt.figure(figsize=(12.0, 8.0))
            # plt.plot(range(0, len(X)), X)
            # plt.xlabel('Sample')
            # plt.ylabel('RSSI (dBm)')
            # plt.ylim(-75, -25)
            # # plt.title('Raw data from ap %s for %s' %
            # #           (ap, TRILATERATION['macs'][mac]))
            # plt.show()

            # Plot filtered data
            # plt.figure()
            # plt.plot(filtered_X, range(0, len(X)))
            # plt.xlabel('RSSI (dBm)')
            # plt.ylabel('Sample')
            # plt.xlim(-80, -20)
            # plt.title('Filtered data from ap %s for %s' %
            #           (ap, TRILATERATION['macs'][mac]))

            time_ranges = []
            for i in distances:
                start_timestamp = convert_date_to_secs(
                    CURVES[str(i)+'m_start'])
                end_timestamp = convert_date_to_secs(CURVES[str(i)+'m_end'])
                time_ranges.append((start_timestamp, end_timestamp))

            # Compute average RSSI for every distance
            data = []
            hist_data = []
            avgs = []
            medians = []
            i = 1
            for s, e in time_ranges:
                single_data = []
                for p in t:
                    if p in range(s, e):
                        single_data.append(X[t.index(p)])
                data = data + list(zip(single_data, [i]*len(single_data)))
                hist_data.append(single_data)
                i += 2
                medians.append(median(single_data))
                avgs.append(round(mean(single_data), 1))
            # all_data.append(data)
            all_avgs.append(avgs)
            all_medians.append(medians)

            # # Color-coded histogram
            # plt.figure(figsize=(12.0, 8.0))
            # plt.hist(hist_data, bins=30, label=distances)
            # plt.xlabel('RSSI (dBm)')
            # plt.ylabel('Frequency')
            # plt.legend(loc='upper right')
            # plt.xlim(-75, -25)
            # # plt.title('RSSI frequencies as detected by ap %s from %s at different distances' %
            # #           (ap, TRILATERATION['macs'][mac]))

            # # Plot collective RSSI histogram
            # plt.figure()
            # plt.hist(filtered_X, bins=30,
            #          density=True, histtype='step')
            # plt.xlabel('RSSI (dB)')
            # plt.ylabel('Frequency')
            # # plt.title('Collective RSSI from ap %s for %s' %
            # #           (ap, TRILATERATION['macs'][mac]))

            # # # Kernel Density Estimation
            # X = np.array(filtered_X)
            # # bandwidths = 10 ** np.linspace(-1, 1, 100)
            # # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
            # #                     {'bandwidth': bandwidths},
            # #                     cv=LeaveOneOut.get_n_splits(X))
            # # grid.fit(X[:, None])
            # # bandwidth = grid.best_params_['bandwidth']
            # bandwidth = 1.5
            # X_d = np.linspace(-75, -25, X.shape[0])
            # kde = KernelDensity(
            #     bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
            # logprob = kde.score_samples(X_d[:, None])
            # plt.fill_between(X_d, np.exp(logprob), alpha=0.2, color='b')
            # plt.xlabel('RSSI (dBm)')
            # plt.ylabel('Probability Density')
            # # plt.legend(loc='upper right')
            # # plt.title('Gaussian kernel estimation of RSSI data from ap %d' % (ap))

            # # # Plot averaged data
            # # plt.figure()
            # # plt.plot(avgs, distances, label='averaged RSSI')

            # # # Fit curve
            # # popt, _ = curve_fit(log_dist_func, avgs, distances)

            # # # Plot curve
            # # avgs.sort()
            # # print('RSSI averages for ap %s and device %s: %s' %
            # #       (ap, TRILATERATION['macs'][mac], avgs))
            # # plt.plot(avgs, log_dist_func(avgs, *popt), 'g--',
            # #          label='fit: $d_0$=-%5.3f, $\gamma$=%5.3f' % tuple(popt))
            # # plt.xlabel('RSSI (dBm)')
            # # plt.ylabel('Distance (m)')
            # # # plt.title('Curve fit for ap %s and device %s' %
            # # #           (ap, TRILATERATION['macs'][mac]))
            # # plt.ylim(0, 12)
            # # plt.legend(loc='upper right')
            # plt.show()
    avged_avgs = []
    for i in np.array(all_medians).T:
        avged_avgs.append(mean(i))

    plt.figure(figsize=(12.0, 8.0))
    r, _ = zip(*data)
    plt.hist(r, bins=300, density=True)
    plt.xlim(-100, 0)

    plt.figure(figsize=(12.0, 8.0))
    # plt.plot(avged_avgs, distances, label='mean RSSI')
    # x, y = zip(*sorted(zip(avged_avgs, distances)))
    x, y = zip(*sorted(data, key=lambda x: x[0]))
    plt.scatter(y, x)
    popt, _ = curve_fit(log_dist_func, x, y)

    # Plot curve
    print('Collective RSSI averages: %s' % avged_avgs)
    x_fit = np.linspace(0, -100, 100)
    plt.plot(log_dist_func(x_fit, *popt), x_fit, 'g--',
             label='fit: $d_0$=-%5.3f, $\gamma$=%5.3f' % tuple(popt))
    plt.ylabel('RSSI (dBm)')
    plt.xlabel('Distance (m)')
    # plt.title('Ultimate curve fit')
    plt.xlim(0, 12)
    plt.ylim(-90, 0)
    plt.legend(loc='upper right')
    plt.show()


def fit_all():
    """
    Plots from heterogenity experiment
    """

    # Plotting the histograms of the different phones on every ap
    for ap in CURVES['aps']:
        plt.figure()
        for mac in CURVES['macs']:
            # _, x1, _ = get_rss_fluctuation(
            #     CURVES['1m_start'], CURVES['5m_end'], ap, mac)
            # _, x2, _ = get_rss_fluctuation(
            #     CURVES['7m_start'], CURVES['11m_end'], ap, mac)
            # X = x1 + x2
            _, X, _ = get_rss_fluctuation(
                CURVES['7m_start'], CURVES['7m_end'], ap, mac)

            # Apply Kalman filter
            kalman = KalmanFilter(0.01, 0.1)
            filtered_X = []
            for p in X:
                filtered_X.append(kalman.filter(int(p)))
            X = filtered_X

            # Data Histogram
            # plt.figure()
            # plt.hist(X, bins=30, label=mac, histtype='step', density=True)
            # plt.xlabel('RSSI (dBm)')
            # plt.ylabel('Frequency')
            # plt.legend()
            # plt.title('Collective RSSI for ap %d' % ap)

            # Kernel Density Estimation
            X = np.array(X)
            # bandwidths = 10 ** np.linspace(-1, 1, 100)
            # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
            #                     {'bandwidth': bandwidths},
            #                     cv=LeaveOneOut().get_n_splits(X))
            # grid.fit(X[:, None])
            # bandwidth = grid.best_params_['bandwidth']
            # print(grid.best_params_)
            bandwidth = 3.0
            X_d = np.linspace(-90, -10, X.shape[0])
            kde = KernelDensity(
                bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
            logprob = kde.score_samples(X_d[:, None])
            plt.fill_between(X_d, np.exp(logprob), alpha=0.2,
                             label=TRILATERATION['macs'][mac])
            plt.xlabel('RSSI (dBm)')
            plt.ylabel('Probability Density')
            plt.legend()
            # plt.title('Kernel estimation for ap %d with bandwidth=%.2f' %
                    #   (ap, bandwidth))
    plt.show()

    # Plotting the histograms of the different aps on every phone
    for mac in CURVES['macs']:
        plt.figure()
        for ap in CURVES['aps']:
            # _, x1, _ = get_rss_fluctuation(
            #     CURVES['1m_start'], CURVES['5m_end'], ap, mac)
            # _, x2, _ = get_rss_fluctuation(
            #     CURVES['7m_start'], CURVES['11m_end'], ap, mac)
            # X = x1 + x2
            _, X, _ = get_rss_fluctuation(
                CURVES['7m_start'], CURVES['7m_end'], ap, mac)

            # Apply Kalman filter
            kalman = KalmanFilter(0.01, 0.1)
            filtered_X = []
            for p in X:
                filtered_X.append(kalman.filter(int(p)))
            X = filtered_X

            # Data Histogram
            # plt.hist(X, bins=30, label=ap, histtype='step', stacked=True)
            # plt.xlabel('RSSI (dB)')
            # plt.ylabel('Frequency')
            # plt.legend()
            # plt.title('Collective RSSI for phone %s' % mac)

            # Kernel Density Estimation
            X = np.array(X)
            # bandwidths = 10 ** np.linspace(-1, 1, 100)
            # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
            #                     {'bandwidth': bandwidths},
            #                     cv=LeaveOneOut().get_n_splits(X))
            # grid.fit(X[:, None])
            # bandwidth = grid.best_params_['bandwidth']
            # print(grid.best_params_)
            bandwidth = 3.0
            X_d = np.linspace(-90, -10, X.shape[0])
            kde = KernelDensity(
                bandwidth=bandwidth, kernel='gaussian').fit(X[:, None])
            logprob = kde.score_samples(X_d[:, None])
            plt.fill_between(X_d, np.exp(logprob), alpha=0.2,
                            label='AP ' + str(ap))
            plt.xlabel('RSSI (dBm)')
            plt.ylabel('Probability Density')
            plt.legend()
            # plt.title('Kernel estimation for device %s with bandwidth=%.2f' %
            #           (mac, bandwidth))
    plt.show()


def heterogeneity_scatter():
    distances = range(1, 12, 2)
    min_num_samples = 100
    mac_combs = combinations(CURVES['macs'], 2)
    for mac1, mac2 in mac_combs:
        for ap in CURVES['aps']:
            x1_all = []
            x2_all = []
            x1_avgs = []
            x2_avgs = []
            for m in distances:
                _, X1, _ = get_rss_fluctuation(
                    CURVES[str(m)+'m_start'], CURVES[str(m)+'m_end'], ap, mac1)
                _, X2, _ = get_rss_fluctuation(
                    CURVES[str(m)+'m_start'], CURVES[str(m)+'m_end'], ap, mac2)
                if len(X1) < min_num_samples:
                    min_num_samples = len(X1)
                if len(X2) < min_num_samples:
                    min_num_samples = len(X2)
                x1_all.append(X1)
                x2_all.append(X2)
                x1_avgs.append(mean(X1))
                x2_avgs.append(mean(X2))

            x1_avgs, x2_avgs = zip(*sorted(zip(x1_avgs, x2_avgs)))
            popt, _ = curve_fit(quad_func, x1_avgs, x2_avgs)
            plt.figure()
            plt.plot(range(-75, -25, 5), quad_func(range(-75, -25, 5), *popt))
            # plt.scatter(x1_avgs, x2_avgs, color='r', alpha=0.5)

            x1_chosen = []
            x2_chosen = []
            for x in x1_all:
                x1_chosen.append(np.random.choice(x, min_num_samples))
            for x in x2_all:
                x2_chosen.append(np.random.choice(x, min_num_samples))
            colors = ("red", "orange", "green", "blue", "indigo", "violet")
            for x, y, color, group in zip(x1_chosen, x2_chosen, colors, distances):
                plt.scatter(x, y, c=color, label=group, alpha=0.5)
            # plt.title('Scatter plot for ap %i' % ap)
            plt.xlabel('RSSI from %s (dBm)' % TRILATERATION['macs'][mac1])
            plt.ylabel('RSSI from %s (dBm)' % TRILATERATION['macs'][mac2])
            plt.xlim(-75, -25)
            plt.ylim(-75, -25)
            plt.legend()
        plt.show()
