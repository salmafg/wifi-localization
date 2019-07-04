import numpy as np
import math
import itertools
from operator import itemgetter
from utils import slope, angle
from config import TRILATERATION


def compute_all(loc, r):
    """
    Compute GDOP for all combinations of 3 APs
    """
    combinations = itertools.combinations(r.keys(), 3)
    gdops = []
    for c in combinations:
        gdops.append((c, compute(loc, {k: r[k] for k in c})))
    return gdops


def compute(loc, r):
    A = np.ones((len(r), 3))
    i = 0
    for ap, ri in r.items():
        xy = next(item for item in TRILATERATION['aps']
                  if item['id'] == int(ap))['xy']
        ex = (loc[0] - xy[0]) / ri
        ey = (loc[1] - xy[1]) / ri
        A[i, 0:2] = [ex, ey]
        i += 1
    Q = np.dot(A.T, A)
    Q = np.linalg.inv(Q)
    pdop = np.sqrt(Q[0, 0]**2 + Q[1, 1]**2)
    tdop = np.sqrt(Q[2, 2]**2)
    gdop = np.sqrt(pdop**2 + tdop**2)
    return gdop


# def compute(loc, r):
#     """
#     Compute GDOP for a single combination
#     """

#     # Compute slopes
#     slopes = []
#     for ap, _ in r.items():
#         xy = next(item for item in TRILATERATION['aps']
#                   if item['id'] == int(ap))['xy']
#         m = slope(loc, xy)
#         slopes.append(m)
#     line_combs = itertools.combinations(slopes, 2)

#     # Compute angles
#     angles = []
#     for m1, m2 in line_combs:
#         angles.append(angle(m1, m2))

#     # Compute GDOP
#     gdop = math.sqrt(
#         3 / (math.sin(angles[0])**2 + math.sin(angles[1])**2 + math.sin(angles[2])**2))
#     return gdop


def get_best_combination(combinations):
    """
    Get the AP combination with the minimum GDOP
    """
    return list(min(combinations, key=itemgetter(1)))
