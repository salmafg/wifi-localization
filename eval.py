from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import distance

matplotlib.rcParams.update({'font.size': 20})


def plot_localization_error():
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
    df = pd.read_csv("data/eval/trilat.csv")
    df['Error in meters'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    print(df)
    # print('Mean error in meters: %.2fm' % mean(df['Error in meters']))
    # print('Min. error in meters: %.2fm' % min(df['Error in meters']))
    # print('Max. error in meters: %.2fm' % max(df['Error in meters']))
    print(df['Error in meters'][df['Phone'] == 'nikos'])
    print(df['Error in meters'][df['Phone'] == 'tiny phone'])
    print(df['Error in meters'][df['Phone'] == 'samsung'])
    sns.boxplot(x='Location', y='Error in meters',
                data=df, hue='Phone', showfliers=False)
    plt.figure()
    sns.kdeplot(df['Error in meters'][df['Phone'] == 'samsung'],
                label='samsung', cumulative=True)
    sns.kdeplot(df['Error in meters'][df['Phone'] == 'nikos'],
                label='nikos', cumulative=True)
    sns.kdeplot(df['Error in meters'][df['Phone'] == 'tiny phone'],
                label='tiny phone', cumulative=True)
    plt.xlim(0, 13)
    plt.legend()
    plt.figure()
    plt.hist(df['Error in meters'][df['Phone'] == 'tiny phone'],
             histtype='step', cumulative=True, density=True, bins=1000, label='tiny phone')
    plt.hist(df['Error in meters'][df['Phone'] == 'samsung'],
             histtype='step', cumulative=True, density=True, bins=1000, label='samsung')
    plt.hist(df['Error in meters'][df['Phone'] == 'nikos'],
             histtype='step', cumulative=True, density=True, bins=1000, label='nikos')
    plt.xlabel('Error in meters')
    plt.ylabel('CDF')
    plt.xlim(0, 13)
    plt.legend()
    plt.show()


def point_of_failure():
    # Point of failure experiment
    # location: (1.0, 7.0)
    # phone: samsung, tiny, george
    # start: 16 Sep 2019 17:35
    # end: 16 Sep 2019 17:45
    df = pd.read_csv("data/eval/pof.csv")
    df['Error in meters'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    print(df)
    print('Mean error in meters: %.2fm' % mean(df['Error in meters']))
    print('Min. error in meters: %.2fm' % min(df['Error in meters']))
    print('Max. error in meters: %.2fm' % max(df['Error in meters']))
    sns.boxplot(x=32-df['Connected access points'],
                y='Error in meters', data=df, order=range(32, 2, -1))
    # plt.figure()
    # for i in range(0, 29):
    #     sns.kdeplot(df['Error in meters'][df['Turned off'] == i],
    #                 label=32-i, cumulative=True)
    # plt.xlabel('Error in meters')
    # plt.ylabel('CDF')
    plt.show()
