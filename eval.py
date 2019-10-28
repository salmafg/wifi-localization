from statistics import mean, median

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score

from fit_data import quad_func
from utils import closest_access_points, distance

matplotlib.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'xtick.labelsize': 'xx-small',
    'ytick.labelsize': 'xx-small',
    'legend.fontsize': 'xx-small',
    'figure.autolayout': True
})


def localization_error(filename):
    # Localization error experiment
    # Location 1: middle of corridor
    # Location 2: room 53
    # Location 3: room 56
    # Location 4: room 65
    # Location 5: end of corridor
    # Location 6: room 55
    # Location 7: room 59
    # Location 8: room 62
    # Location 9: room 51
    # Location 10: beginnning of corridor
    df = pd.read_csv(filename)
    # df['Location'] = 1
    # df['Phone'] = 'george'
    df['Error in meters no polygons'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    df['Error in meters with polygons'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['new_x'], row['new_y'])), axis=1)
    print('Case: %s' % filename)
    print('Mean error (m) no polygons: %.3fm' %
          mean(df['Error in meters no polygons']))
    print('Median error (m) no polygons: %.3fm' %
          median(df['Error in meters no polygons']))
    # print('Min. error (m) no polygons: %.3fm' %
    #       min(df['Error in meters no polygons']))
    # print('Max. error (m) no polygons: %.3fm' %
    #       max(df['Error in meters no polygons']))
    print('Mean error (m) with polygons: %.3fm' %
          mean(df['Error in meters with polygons']))
    print('Median error (m) with polygons: %.3fm' %
          median(df['Error in meters with polygons']))
    # print('Min. error (m) with polygons: %.3fm' %
    #       min(df['Error in meters with polygons']))
    # print('Max. error (m) with polygons: %.3fm' %
    #       max(df['Error in meters with polygons']))
    plt.figure(figsize=(12.0, 8.0))
    sns.boxplot(x='Location', y='Error in meters no polygons',
                data=df, showfliers=True, color="skyblue")
    plt.ylabel('Error (m)')
    # plt.figure()
    # sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'samsung'],
    #             label='samsung', cumulative=True, label='Phone A')
    # sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'nikos'],
    #             label='nikos', cumulative=True, label='Phone D')
    # sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'tiny phone'],
    #             label='tiny phone', cumulative=True, label='Phone B')
    # sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'george'],
    #             label='george', cumulative=True, label='Phone C')
    # sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'lg'],
    #             label='lg', cumulative=True, label='Phone E')
    # plt.xlabel('Error (m)')
    # plt.ylabel('CDF')
    # plt.xlim(0, 18)
    # plt.legend()
    plt.figure(figsize=(12.0, 8.0))
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'samsung'],
             histtype='step', cumulative=True, density=True, bins=1000, label='Phone A')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'tiny phone'],
             histtype='step', cumulative=True, density=True, bins=1000, label='Phone B')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'george'],
             histtype='step', cumulative=True, density=True, bins=1000, label='Phone C')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'nikos'],
             histtype='step', cumulative=True, density=True, bins=1000, label='Phone D')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'lg'],
             histtype='step', cumulative=True, density=True, bins=1000, label='Phone E')
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.xlim(0, 7.6)
    plt.legend(loc='lower right')
    plt.show()
    before = accuracy_score(df['true_polygon'][df['No polygons'] == 'unknown'],
                            df['No polygons'][df['No polygons'] == 'unknown'])
    after = accuracy_score(df['true_polygon'][df['No polygons'] == 'unknown'],
                           df['Polygons'][df['No polygons'] == 'unknown'])
    print('%.1f%% improvement' % (100*after-100*before))


def point_of_failure(filename):
    # Point of failure experiment
    # location: (1.0, 7.0)
    # phone: samsung, tiny, george
    # start: 16 Sep 2019 17:35
    # end: 16 Sep 2019 17:45
    df = pd.read_csv(filename)
    df['Error in meters'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    df['Error'] = df['obs_y'] - df['true_y']
    # sns.boxplot(x='Connected', y='Error', data=df, order=range(29, 2, -1))
    # print(df)
    sns.scatterplot(x='obs_x', y='obs_y', sizes=2,
                    hue_norm=(3, 29), hue='Connected', data=df)
    plt.scatter(df['true_x'][0], df['true_y'][0],
                c='black', marker='x', alpha=1)
    plt.xlabel('Predicted x location')
    plt.ylabel('Predicted y location')
    # plt.scatter(df['obs_x'], df['obs_y'], label=df['Connected'])
    print('Mean error in meters: %.2fm' % mean(df['Error in meters']))
    print('Min. error in meters: %.2fm' % min(df['Error in meters']))
    print('Max. error in meters: %.2fm' % max(df['Error in meters']))
    plt.figure(figsize=(12.0, 8.0))
    ax = sns.boxplot(x='Connected', y='Error in meters',
                     data=df, order=range(29, 2, -1))
    plt.xlabel('Number of Connected APs')
    plt.ylabel('Error (m)')
    ax2 = ax.twiny()
    distances = [round(y, 3) for (x, y) in closest_access_points(
        (df['true_x'][0], df['true_y'][0]))]
    print(distances)
    ax2.xaxis.set_ticks(np.arange(0.5, 27.5, 1))
    ax2.xaxis.set_ticklabels(distances)
    ax2.set_xlabel('Distance to Closest AP (m)')
    plt.xticks(rotation=50)
    # plt.figure()
    # for i in range(0, 29):
    #     sns.kdeplot(df['Error in meters'][df['Connected'] == i],
    #                 label=32-i, cumulative=True)
    # plt.xlabel('Error (m)')
    # plt.ylabel('CDF')
    plt.show()


def uncertainty(filename):
    df = pd.read_csv(filename)
    df['Deviation from radius'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])) - row['Uncertainty'], axis=1)
    df['Location'] = 1
    print(df.head())
    print('Mean deviation in meters: %.2fm' %
          mean(df['Deviation from radius'][df['Threshold'] == -85]))
    print('Mean deviation in meters: %.2fm' %
          median(df['Deviation from radius'][(df['Threshold'] == -85) & (df['Deviation from radius'] >= 0)]))
    print('Median deviation in meters: %.2fm' %
          median(df['Deviation from radius'][df['Threshold'] == -85]))
    print('Min. deviation in meters: %.2fm' % min(df['Deviation from radius']))
    print('Max. deviation in meters: %.2fm' % max(df['Deviation from radius']))
    # sns.boxplot(x='Location', y='Deviation from radius',
    #             data=df, showfliers=True)
    # plt.ylabel('Deviation from uncertainty radius (m)')
    plt.figure(figsize=(12.0, 8.0))
    sns.kdeplot(df['Deviation from radius'][(df['Threshold'] == -85) & (df['Phone'] == 'samsung')],
                label='Phone A', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'tiny'],
                label='Phone B', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'george'],
                label='Phone C', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'nikos'],
                label='Phone D', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'lg'],
                label='Phone E', cumulative=True)
    plt.xlabel('Deviation from Uncertainty Perimeter (m)')
    plt.ylabel('CDF')
    plt.xlim(-10, 10)
    plt.legend()
    plt.show()


def rssi_threshold(filename):

    # RSSI threshold experiment
    df = pd.read_csv(filename)
    df['Error in meters'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    # print(df)
    location_filter = (df[df.Location == 1])
    thresholds = location_filter.groupby('Threshold').size()
    _, ax = plt.subplots()
    ax.set_xlabel('RSSI Threshold (dB)')
    ax.set_ylabel('Localization Frequency')
    for i in range(1, 6, 1):
        ax.plot((df[df.Location == i]).groupby(
            'Threshold').size(), label=i, linestyle='--')
    # plt.legend()
    # ax2 = ax.twinx()
    # ax2.plot(location_filter.groupby('Threshold')
    #          ['Error in meters'].agg(['mean']), c='g')
    # ax2.set_ylabel('Error (m)', c='g')
    # sns.boxplot(x='Threshold', y='Error in meters', hue='Location', data=df)
    plt.figure()
    for i in range(-85, -40, 10):
        # sns.kdeplot(df['Error in meters'][df['Threshold'] == i],
        #             label=i, cumulative=True)
        plt.hist(df['Error in meters'][df['Threshold'] == i],
                 histtype = 'step', cumulative = True, density = True, bins = 1000, label = i)
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.xlim(0, 6)
    plt.legend(loc = 'lower right')
    plt.show()


def distance_estimation(dlos, nlos):
    ddf=pd.read_csv(dlos)
    # ddf = ddf[(ddf['phone'] == 'lg') | (ddf['phone'] == 'nikos')]
    ddf.groupby('true_distance')[
        'estimated_distance'].mean().plot(label='DLOS')
    ndf = pd.read_csv(nlos)
    # ndf = ndf[(ndf['phone'] == 'lg') | (ndf['phone'] == 'nikos')]
    ndf.groupby('true_distance')[
        'estimated_distance'].mean().plot(label='NLOS')
    plt.rc('axes', axisbelow=True)
    plt.xlabel('True distance (m)')
    plt.ylabel('Avg. estimated distance (m)')
    plt.xticks(range(1, 12, 2))
    plt.yticks(range(1, 12, 2))
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()


def dubai_data(filename):
    df = pd.read_csv(filename)
    df = pd.melt(df, id_vars=['timestamp', 'longitude', 'latitude'], value_vars=['6', '2', '8', '3', '26', '15', '12',
                                                                                 '5', '13', '27', '25', '22'], var_name='sensor_id', value_name='rssi')
    df['mac'] = '85362b02a0d505c7a3a9aca24e9e480778082adb242ff6dfb49b6acc62375cbb'
    df.to_json('dubai_data.json', orient='records')

# Localization error experiment: tiny, george, samsung, nikos
# Location 1: middle of corridor (1.0, 7.0), 4 Oct 2019 16:32 - 16:42
# Location 2: room 53 (-2.0, 5.1), 4 Oct 2019 16:43:10 - 16:53
# Location 3.1: room 56 (3.6, 9.5), 4 Oct 2019 16:54:10 - 17:04
# Location 3.2: room 56 (2.342, 8.97), 10 Oct 2019 17:10:00 - 17:21
# Location 4: room 65 (-2.6, 27.0), 4 Oct 2019 17:05:10 - 17:15
# Location 5.1: end of corridor (0.0, 29.0), 4 Oct 2019 17:16:10 - 17:26 #!1237
# Location 5.2: end of corridor (0.5, 22.5), 7 Oct 2019 14:18:00 - 14:28 #!244,246,314
# Location 6: room 55 (-2.8, 9.55), 4 Oct 2019 17:27:10 - 17:37
# Location 7: room 59 (-2.0, 17.0), 4 Oct 2019 17:38:15 - 17:48
# Location 8: room 62 (3.7, 19.2), 4 Oct 2019 17:49:10 - 17:59 #!george
# Location 9: room 51 (-2.4, 1.76), 4 Oct 2019 18:00:10 - 18:10
# Location 10: beginnning of corridor (0.0, 1.8), 4 Oct 2019 18:11:10 - 18:21

# Localization error experiment: LG
# Location 1: middle of corridor (1.0, 7.0), 11 Oct 2019 11:22 - 11:32
# Location 2: room 53 (-2.0, 5.1), 11 Oct 2019 11:32:10 - 11:42
# Location 3.1: room 56 (3.6, 9.5), 11 Oct 2019 11:42:10 - 11:52
# Location 4: room 65 (-2.6, 27.0), 11 Oct 2019 12:02:10 - 12:12
# Location 5.2: end of corridor (0.5, 22.5), 11 Oct 2019 11:52:10 - 12:02
# Location 6: room 55 (-2.8, 9.55), 11 Oct 2019 12:12:10 - 12:22
# Location 7: room 59 (-2.0, 17.0), 11 Oct 2019 12:22:10 - 12:32
# Location 8: room 62 (3.7, 19.2), 11 Oct 2019 12:32:10 - 12:42
# Location 9: room 51 (-2.4, 1.76), 11 Oct 2019 12:42:10 - 12:52
# Location 10: beginnning of corridor (0.0, 1.8), 11 Oct 2019 12:52:10 - 13:02

# POF 2:
# location: (0.875, 15.0)
# phone: samsung, tiny, lg
# start: 11 Oct 2019 13:38
# end: 11 Oct 2019 13:50

# Distance estimation:
# phones: samsung, george, nikos, tiny, lg
# aps: 43, 21, 39, 9, 19
# DLOS:
# 1m: 16 Oct 2019 13:38:00 - 13:48:00
# 3m: 16 Oct 2019 13:48:10 - 13:58:00
# 5m: 16 Oct 2019 13:58:40 - 14:08:00
# 7m: 16 Oct 2019 14:08:10 - 14:18:00
# 9m: 16 Oct 2019 14:18:10 - 14:28:00
# 11m: 16 Oct 2019 14:28:10 - 14:38:00
# NLOS:
# 1m: 16 Oct 2019 14:52:00 - 15:02:00
# 3m: 16 Oct 2019 15:02:10 - 15:12:00
# 5m: 16 Oct 2019 15:12:10 - 15:22:00
# 7m: 16 Oct 2019 15:22:10 - 15:32:00
# 9m: 16 Oct 2019 15:32:10 - 15:42:00
# 11m: 16 Oct 2019 15:42:10 - 15:52:00
