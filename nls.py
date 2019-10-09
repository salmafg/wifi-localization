import math

from scipy.optimize import least_squares

from utils import distance


def residuals(guess, p, r):
    x, y = guess
    res = ()
    for i in p:
        xi = p[i][0]
        yi = p[i][1]
        ri = r[i]
        # res += ((distance((x, y), (xi, yi)) - ri), ) # nls
        # res += ((distance((x, y), (xi, yi)) - ri) / ri, ) # dist
        res += ((distance((x, y), (xi, yi)) - ri) / ri**2, ) # var
    return res


def nls(guess, p, r):
    ls = least_squares(residuals, guess, args=(p, r))
    return ls.x
