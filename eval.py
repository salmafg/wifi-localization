from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from utils import distance

matplotlib.rcParams.update({
    'font.size': 22,
    'font.family': 'serif',
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'x-small',
    'legend.fontsize': 'xx-small',
    'figure.autolayout': True
})

def plot_localization_error(filename):
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
    df['Error in meters no polygons'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    df['Error in meters with polygons'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['new_x'], row['new_y'])), axis=1)
    print('Case: %s' % filename)
    print('Mean error (m) no polygons: %.3fm' %
          mean(df['Error in meters no polygons']))
    print('Min. error (m) no polygons: %.3fm' %
          min(df['Error in meters no polygons']))
    print('Max. error (m) no polygons: %.3fm' %
          max(df['Error in meters no polygons']))
    print('Mean error (m) with polygons: %.3fm' %
          mean(df['Error in meters with polygons']))
    print('Min. error (m) with polygons: %.3fm' %
          min(df['Error in meters with polygons']))
    print('Max. error (m) with polygons: %.3fm' %
          max(df['Error in meters with polygons']))
    sns.boxplot(x='Location', y='Error in meters no polygons',
                data=df, showfliers=True, color="skyblue")
    plt.ylabel('Error (m)')
    plt.figure()
    sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'samsung'],
                label='samsung', cumulative=True)
    sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'nikos'],
                label='nikos', cumulative=True)
    sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'tiny phone'],
                label='tiny phone', cumulative=True)
    sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'george'],
                label='george', cumulative=True)
    sns.kdeplot(df['Error in meters no polygons'][df['Phone'] == 'lg'],
                label='lg', cumulative=True)
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.xlim(0, 18)
    plt.legend()
    plt.figure()
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'tiny phone'],
             histtype='step', cumulative=True, density=True, bins=1000, label='tiny phone')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'samsung'],
             histtype='step', cumulative=True, density=True, bins=1000, label='samsung')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'nikos'],
             histtype='step', cumulative=True, density=True, bins=1000, label='nikos')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'george'],
             histtype='step', cumulative=True, density=True, bins=1000, label='george')
    plt.hist(df['Error in meters no polygons'][df['Phone'] == 'lg'],
             histtype='step', cumulative=True, density=True, bins=1000, label='lg')
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.xlim(0, 15)
    plt.legend()
    plt.show()
    before = accuracy_score(df['true_polygon'][df['No polygons'] == 'unknown'],
                            df['No polygons'][df['No polygons'] == 'unknown'])
    after = accuracy_score(df['true_polygon'][df['No polygons'] == 'unknown'],
                           df['Polygons'][df['No polygons'] == 'unknown'])
    # print('Before applying polygons:', before)
    # print('After applying polygons:', after)
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
    # print(df)
    print('Mean error in meters: %.2fm' % mean(df['Error in meters']))
    print('Min. error in meters: %.2fm' % min(df['Error in meters']))
    print('Max. error in meters: %.2fm' % max(df['Error in meters']))
    sns.boxplot(x='Closest', y='Error in meters', data=df)
    plt.xlabel('Closest access point (m)')
    plt.ylabel('Error (m)')
    # plt.figure()
    # for i in range(0, 29):
    #     sns.kdeplot(df['Error in meters'][df['Connected'] == i],
    #                 label=32-i, cumulative=True)
    # plt.xlabel('Error (m)')
    # plt.ylabel('CDF')
    plt.show()


def eval_uncertainty(filename):
    df = pd.read_csv(filename)
    df['Deviation from radius'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])) - row['Uncertainty'], axis=1)
    # print(df.head())
    print('Mean deviation in meters: %.2fm' %
          mean(df['Deviation from radius']))
    print('Min. deviation in meters: %.2fm' % min(df['Deviation from radius']))
    print('Max. deviation in meters: %.2fm' % max(df['Deviation from radius']))
    sns.boxplot(x='Location', y='Deviation from radius',
                data=df, showfliers=True)
    plt.ylabel('Deviation from uncertainty radius (m)')
    plt.figure()
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'samsung'],
                label='samsung', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'george'],
                label='george', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'tiny phone'],
                label='tiny phone', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'nikos'],
                label='nikos', cumulative=True)
    plt.xlabel('Deviation from uncertainty radius (m)')
    plt.ylabel('CDF')
    plt.xlim(-20, 20)
    plt.legend()
    plt.show()


def rssi_threshold(filename):
    # RSSI threshold experiment
    df = pd.read_csv(filename)
    df['Error in meters'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    # print(df)
    sns.boxplot(x='Threshold', y='Error in meters', data=df)
    # plt.figure()
    # for i in range(0, 29):
    #     sns.kdeplot(df['Error in meters'][df['Disconnected'] == i],
    #                 label=32-i, cumulative=True)
    # plt.xlabel('Error (m)')
    # plt.ylabel('CDF')
    plt.show()


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
# Location 8: room 62 (3.7, 19.2), 4 Oct 2019 17:49:10 - 17:59
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
