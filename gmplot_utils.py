import sys

import gmplot

from config import BUILDING, GMPLOT
from mi import MAP

GMAP = gmplot.GoogleMapPlotter(GMPLOT['lat'], GMPLOT['lng'], GMPLOT['zoom'])
GMAP.apikey = GMPLOT['apiKey']


def main():
    if len(sys.argv) == 3:
        plot_point()
    # plot_building()
    # plot_corridor()
    plot_room(['00.11.065', '00.11.062', '00.11.059', '00.11.056', '00.11.055',
               '00.11.054', '00.11.053', '00.11.051', 'the corridor'])


def plot_point():
    args = [float(x) for x in sys.argv[1:]]
    GMAP.scatter([args[0]], [args[1]], '#3B0B3', size=0.5, marker=False)
    GMAP.draw('./map.html')


def plot_building():
    lats = [lat for (lat, lng) in BUILDING]
    lngs = [lng for (lat, lng) in BUILDING]
    GMAP.polygon(lats, lngs, color='cornflowerblue')
    GMAP.draw('./map.html')


def plot_corridor():
    corridor = next(d for (index, d) in enumerate(
        MAP) if d['properties']['ref'] == "the corridor")['geometry']['coordinates']
    lats = [lat for (lng, lat) in corridor]
    lngs = [lng for (lng, lat) in corridor]
    GMAP.polygon(lats, lngs, color='cornflowerblue')
    GMAP.draw('./map.html')


def plot_room(room_nums):
    for r in room_nums:
        room = next(d for (index, d) in enumerate(
            MAP) if d['properties']['ref'] == r)['geometry']['coordinates']
        lats = [lat+0.00003 for (lng, lat) in room]
        lngs = [lng+0.00004 for (lng, lat) in room]
        GMAP.polygon(lats, lngs, color='cornflowerblue')
    GMAP.draw('./map.html')


if __name__ == '__main__':
    main()
