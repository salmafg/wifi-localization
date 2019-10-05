from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

from utils import distance

matplotlib.rcParams.update({'font.size': 20})


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
    df['Error in meters'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    df['Error in meters with polygons'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['new_x'], row['new_y'])), axis=1)
    print('Mean error in meters: %.3fm' % mean(df['Error in meters']))
    print('Min. error in meters: %.3fm' % min(df['Error in meters']))
    print('Max. error in meters: %.3fm' % max(df['Error in meters']))
    print('Mean error in meters: %.3fm' % mean(df['Error in meters with polygons']))
    print('Min. error in meters: %.3fm' % min(df['Error in meters with polygons']))
    print('Max. error in meters: %.3fm' % max(df['Error in meters with polygons']))
    sns.boxplot(x='Location', y='Error in meters', data=df, showfliers=True)
    plt.figure()
    sns.kdeplot(df['Error in meters'][df['Phone'] == 'samsung'],
                label='samsung', cumulative=True)
    sns.kdeplot(df['Error in meters'][df['Phone'] == 'nikos'],
                label='nikos', cumulative=True)
    sns.kdeplot(df['Error in meters'][df['Phone'] == 'tiny phone'],
                label='tiny phone', cumulative=True)
    sns.kdeplot(df['Error in meters'][df['Phone'] == 'george'],
                label='george', cumulative=True)
    plt.xlim(0, 15)
    plt.legend()
    plt.figure()
    plt.hist(df['Error in meters'][df['Phone'] == 'tiny phone'],
             histtype='step', cumulative=True, density=True, bins=1000, label='tiny phone')
    plt.hist(df['Error in meters'][df['Phone'] == 'samsung'],
             histtype='step', cumulative=True, density=True, bins=1000, label='samsung')
    plt.hist(df['Error in meters'][df['Phone'] == 'nikos'],
             histtype='step', cumulative=True, density=True, bins=1000, label='nikos')
    plt.hist(df['Error in meters'][df['Phone'] == 'george'],
             histtype='step', cumulative=True, density=True, bins=1000, label='george')
    plt.xlabel('Error in meters')
    plt.ylabel('CDF')
    plt.xlim(0, 15)
    plt.legend()
    plt.show()
    before = accuracy_score(df['true_polygon'], df['No polygons'])
    after = accuracy_score(df['true_polygon'], df['Polygons'])
    print('Before applying polygons:', before)
    print('After applying polygons:', after)
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
    print(df)
    print('Mean error in meters: %.2fm' % mean(df['Error in meters']))
    print('Min. error in meters: %.2fm' % min(df['Error in meters']))
    print('Max. error in meters: %.2fm' % max(df['Error in meters']))
    sns.boxplot(x=32-df['Disconnected access points'],
                y='Error in meters', data=df, order=range(32, 2, -1))
    plt.xlabel('Number of access points')
    # plt.figure()
    # for i in range(0, 29):
    #     sns.kdeplot(df['Error in meters'][df['Disconnected access points'] == i],
    #                 label=32-i, cumulative=True)
    # plt.xlabel('Error in meters')
    # plt.ylabel('CDF')
    plt.show()


def eval_uncertainty():
    df = pd.read_csv("data/eval/uncertainty.csv")
    df['Deviation from radius'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])) - row['uncertainty'], axis=1)
    print(df.head())
    print('Mean deviation in meters: %.2fm' %
          mean(df['Deviation from radius']))
    print('Min. deviation in meters: %.2fm' % min(df['Deviation from radius']))
    print('Max. deviation in meters: %.2fm' % max(df['Deviation from radius']))
    sns.boxplot(x='Location', y='Deviation from radius',
                data=df, hue='Phone', showfliers=True)
    plt.figure()
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'samsung'],
                label='samsung', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'george'],
                label='george', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'tiny phone'],
                label='tiny phone', cumulative=True)
    plt.legend()
    plt.show()


# Localization error experiment: tiny, george, samsung, nikos
# Location 1: middle of corridor (1.0, 7.0), 4 Oct 2019 16:32 - 16:42
# Location 2: room 53 (-2.0, 5.1), 4 Oct 2019 16:43:10 - 16:53
# Location 3: room 56 (3.6, 9.5), 4 Oct 2019 16:54:10 - 17:04
# Location 4: room 65 (-2.6, 27.0), 4 Oct 2019 17:05:10 - 17:15
# Location 5: end of corridor (0.0, 29.0), 4 Oct 2019 17:16:10 - 17:26 #!1237
# Location 6: room 55 (-2.8, 9.55), 4 Oct 2019 17:27:10 - 17:37
# Location 7: room 59 (-2.0, 17.0), 4 Oct 2019 17:38:15 - 17:48
# Location 8: room 62 (3.7, 19.2), 4 Oct 2019 17:49:10 - 17:59
# Location 9: room 51 (-2.4, 1.76), 4 Oct 2019 18:00:10 - 18:10
# Location 10: beginnning of corridor (0.0, 1.8), 4 Oct 2019 18:11:10 - 18:21
