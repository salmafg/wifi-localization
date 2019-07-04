import numpy as np
import math
import itertools
from operator import itemgetter
from utils import slope, angle
from config import TRILATERATION


def compute_all(loc, p):
    """
    Compute GDOP for all combinations of 3 APs
    """
    combinations = itertools.combinations(p.keys(), 3)
    gdops = []
    for c in combinations:
        gdops.append((c, compute(loc, p)))
    return gdops


def compute(loc, p):
    """
    Compute GDOP for a single combination
    """

    # Compute slopes
    slopes = []
    for ap, xy in p.items():
        m = slope(loc, xy)
        slopes.append(m)
    line_combs = itertools.combinations(slopes, 2)

    # Compute angles
    angles = []
    for m1, m2 in line_combs:
        angles.append(angle(m1, m2))

    # Compute GDOP
    gdop = math.sqrt(
        3 / (math.sin(angles[0])**2 + math.sin(angles[1])**2 + math.sin(angles[2])**2))
    return gdop


def get_best_combination(combinations):
    """
    Get the AP combination with the minimum GDOP
    """
    return list(min(combinations, key=itemgetter(1)))
