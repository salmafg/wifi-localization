import math

import numpy as np

from config import LOG


def fspl(rss):
    """
    http://goo.gl/cGXmDw
    """
    rss = int(round(rss))
    logd = (27.55 - (20 * math.log10(2400)) + abs(rss)) / 20
    d = math.pow(10, logd)
    return d


def itu(rss):
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


def log(rss):
    """
    https://en.wikipedia.org/wiki/Log-distance_path_loss_model
    """
    rss = int(round(rss))
    pl0 = LOG['pl0']
    d0 = LOG['d0']
    gamma = LOG['gamma']
    logdd0 = (abs(rss) - abs(pl0)) / (10 * gamma)
    dd0 = math.pow(10, logdd0)
    d = dd0 * d0
    return d


def parameter_fitting(dict_of_rss):
    gammas = np.arange(2.0, 6.0, 0.1)
    for gamma in gammas:
        print("For gamma = ", gamma)
        dict_of_distances = {}
        for ap, rss in dict_of_rss.items():
            estimated_distance = log(rss, gamma)
            dict_of_distances[ap] = estimated_distance
            print('The estimated distance of the AP %d is %f' %
                  (ap, estimated_distance))
