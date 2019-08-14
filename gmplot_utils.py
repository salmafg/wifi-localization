import sys

import gmplot

from config import BUILDING, GMPLOT
from mi import MAP

gmap = gmplot.GoogleMapPlotter(GMPLOT['lat'], GMPLOT['lng'], GMPLOT['zoom'])
gmap.apikey = GMPLOT['apiKey']


def main():
    if len(sys.argv) == 3:
        plot_point()
    # plot_building()
    plot_corridor()
    # plot_room()


def plot_point():
    args = [float(x) for x in sys.argv[1:]]
    gmap.scatter([args[0]], [args[1]], '#3B0B3', size=0.5, marker=False)
    gmap.draw('./map.html')


def plot_building():
    lats = [lat for (lat, lng) in BUILDING]
    lngs = [lng for (lat, lng) in BUILDING]
    gmap.polygon(lats, lngs, color='cornflowerblue')
    gmap.draw('./map.html')


def plot_corridor():
    corridor = next(d for (index, d) in enumerate(
        MAP) if d['properties']['ref'] == "the corridor")['geometry']['coordinates']
    lats = [lat for (lng, lat) in corridor]
    lngs = [lng for (lng, lat) in corridor]
    gmap.polygon(lats, lngs, color='cornflowerblue')
    gmap.draw('./map.html')


def plot_room():
    room = next(d for (index, d) in enumerate(
        MAP) if d['properties']['ref'] == "00.11.051")['geometry']['coordinates']
    lats = [lat for (lng, lat) in room]
    lngs = [lng for (lng, lat) in room]
    gmap.polygon(lats, lngs, color='cornflowerblue')
    gmap.draw('./map.html')


if __name__ == '__main__':
    main()
