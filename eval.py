import math
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
    'font.size': 25,
    'font.family': 'serif',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'legend.fontsize': 'medium',
    'figure.autolayout': True,
    'figure.figsize': (12, 8)
})


def localization_error(filename):
    df = pd.read_csv(filename)
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
    ax = sns.boxplot(x='Connected', y='Error in meters', data=df)
    # ax = sns.boxplot(x='Connected', y='Error in meters',
    #                  data=df, order=range(29, 2, -1))
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


def point_of_failure_control(f_closest, f_furthest):
    df1 = pd.read_csv(f_closest)
    df1['Connected APs'] = 'Closest 3'
    df2 = pd.read_csv(f_furthest)
    df2['Connected APs'] = 'Furthest 3'
    df = pd.concat([df1, df2])
    df['Error in meters'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    plt.figure(figsize=(12.0, 8.0))
    palette = sns.color_palette("RdBu", 5)
    sns.boxplot(x='Connected APs', y='Error in meters',
                data=df, palette=palette)
    plt.ylabel('Error (m)')
    plt.show()
    print('Median error in meters: %.2fm' %
          median(df['Error in meters'][df['Connected APs'] == 'Closest 3']))
    print('Median error in meters: %.2fm' %
          median(df['Error in meters'][df['Connected APs'] == 'Furthest 3']))


def uncertainty(filename):
    df = pd.read_csv(filename)
    df['Error'] = df.apply(lambda row: distance(
        (row['true_x'], row['true_y']), (row['obs_x'], row['obs_y'])), axis=1)
    df['Location'] = 1
    print(df.head())
    df['Deviation from radius'] = df.apply(lambda row: row['Uncertainty'] - row['Error'], axis=1)
    MAE = mean(abs(df['Uncertainty'] - df['Error']))
    RMSE = ((df['Uncertainty'] - df['Error']) ** 2).mean() ** .5
    print('MAE: %.2fm' % MAE)
    print('RMSE: %.2fm' % RMSE)
    print('Positive deviation in meters: %.2fm' %
          mean(df['Deviation from radius'][(df['Deviation from radius'] > 0)]))
    print('Negative deviation in meters: %.2fm' %
          mean(df['Deviation from radius'][(df['Deviation from radius'] < 0)]))
    print('Min. deviation in meters: %.2fm' % min(df['Deviation from radius']))
    print('Max. deviation in meters: %.2fm' % max(df['Deviation from radius']))
    # sns.boxplot(x='Location', y='Deviation from radius',
    #             data=df, showfliers=True)
    # plt.ylabel('Deviation from uncertainty radius (m)')
    plt.figure(figsize=(12.0, 8.0))
    sns.kdeplot(df['Deviation from radius'][(df['Phone'] == 'samsung')],
                label='Phone A', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'tiny'],
                label='Phone B', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'george'],
                label='Phone C', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'nikos'],
                label='Phone D', cumulative=True)
    sns.kdeplot(df['Deviation from radius'][df['Phone'] == 'lg'],
                label='Phone E', cumulative=True)
    plt.xlabel('Deviation from True Location (m)')
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
    ax.set_xlabel('RSSI Threshold (dBm)')
    ax.set_ylabel('Localization Frequency')
    for i in range(1, 6, 1):
        ax.plot((df[df.Location == i]).groupby(
            'Threshold').size(), label=i, linestyle='--')
    plt.legend(loc='lower left')
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
                 histtype='step', cumulative=True, density=True, bins=1000, label=i)
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.xlim(0, 6)
    plt.legend(loc='lower right')
    plt.show()


def distance_estimation(dlos, nlos):
    plt.figure(figsize=(8, 8))
    ddf = pd.read_csv(dlos)
    ddf['Type'] = 'DLoS'
    ddf.groupby('true_distance')[
        'estimated_distance'].mean().plot(label='DLoS')
    ndf = pd.read_csv(nlos)
    ndf['Type'] = 'NLoS'
    ndf.groupby('true_distance')[
        'estimated_distance'].mean().plot(label='NLoS')
    df = pd.concat([ddf, ndf])
    df['Error'] = df.apply(lambda row: abs(row['true_distance'] - row['estimated_distance']), axis=1)
    plt.rc('axes', axisbelow=True)
    plt.xlabel('True Distance (m)', fontsize='large')
    plt.ylabel('Avg. Estimated Distance (m)', fontsize='large')
    plt.xticks(range(1, 12, 2), fontsize='medium')
    plt.yticks(range(1, 12, 2), fontsize='medium')
    plt.grid(alpha=0.5)
    plt.gca().set_aspect("equal")
    plt.legend(fontsize='medium')
    plt.show()
    # plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.catplot(ax=ax, x='true_distance', y='Error', hue='Type', kind='bar', legend=False, capsize=.1, data=df)
    ax.set_xlabel('True Distance (m)', fontsize='large')
    ax.set_ylabel('MAE (m)', fontsize='large')
    # ax.set_xtickslabels(fontsize='medium')
    ax.tick_params(axis='both', which='major', labelsize=25)
    # ax.set_yticks(fontsize='medium')
    ax.grid(alpha=0.5, axis='y')
    ax.legend(fontsize='medium')
    plt.show()
