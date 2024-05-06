import numpy as np
from datetime import datetime

'''
*This is a python version of "read_select_Swarm_L1b_start.m" for matlab made by Chris Finlay.
'''

#%% ----------------------------------------------- Helping functions -----------------------------------------------

def to_mjd2000(year, month, day, hour=0, minute=0,second=0):
    '''
    converts a date in the input date to Modified Julian Dates(MJD2000), which starts at Jan 1 2000 0h00 and returns the result in decimal days
    '''
    difference_in_days = (datetime(year, month, day, hour, minute, second)
                          - datetime(2000, 1, 1))

    return difference_in_days.days + difference_in_days.seconds / 86400


def revolutions_to_radians(revolutions):
    '''
    Parameters: revolutions in range; 0 <= revolutions <= 1
    Output: Corresponding revolutions in radians; 0 <= radians <= 2*pi
    '''
    return 2 * np.pi * np.mod(revolutions, 1)


def sun_mjd2000(mjd2000_time):
    '''
    Solar emphemeris
    input: modified julian day (MJD2000) time
    output:
        right_ascension: right ascension of the sun [radians] in range; 0 <= right_ascension <= 2 pi
        declination: declination of the sun [radians] in range; -pi/2 <= declination <= pi/2

    Notes:
    coordinates are inertial, geocentric, equatorial and true-of-date

    Ported from MATLAB by Eigil Lippert
    Modified by Nils Olsen, DSRI
    '''
    atr = np.pi / 648000

    # time arguments
    djd = mjd2000_time - 0.5;

    t = (djd / 36525) + 1;

    # fundamental arguments (radians)
    gs = revolutions_to_radians(0.993126 + 0.0027377785 * djd);
    lm = revolutions_to_radians(0.606434 + 0.03660110129 * djd);
    ls = revolutions_to_radians(0.779072 + 0.00273790931 * djd);
    g2 = revolutions_to_radians(0.140023 + 0.00445036173 * djd);
    g4 = revolutions_to_radians(0.053856 + 0.00145561327 * djd);
    g5 = revolutions_to_radians(0.056531 + 0.00023080893 * djd);
    rm = revolutions_to_radians(0.347343 - 0.00014709391 * djd);

    # geocentric, ecliptic longitude of the sun (radians)
    plon = 6910 * np.sin(gs) + 72 * np.sin(2 * gs) - 17 * t * np.sin(gs)
    plon = plon - 7 * np.cos(gs - g5) + 6 * np.sin(lm - ls) + 5 * np.sin(4 * gs - 8 * g4 + 3 * g5)
    plon = plon - 5 * np.cos(2 * (gs - g2)) - 4 * (np.sin(gs - g2) - np.cos(4 * gs - 8 * g4 + 3 * g5));
    plon = plon + 3 * (np.sin(2 * (gs - g2)) - np.sin(g5) - np.sin(2 * (gs - g5)));
    plon = ls + atr * (plon - 17 * np.sin(rm));

    # geocentric distance of the sun (kilometers)
    rsm = 149597870.691 * (1.00014 - 0.01675 * np.cos(gs) - 0.00014 * np.cos(2 * gs));

    # obliquity of the ecliptic (radians)
    obliq = atr * (84428 - 47 * t + 9 * np.cos(rm));

    # geocentric, equatorial right ascension and declination (radians)
    a = np.sin(plon) * np.cos(obliq);
    b = np.cos(plon);

    right_ascension = np.arctan2(a, b);
    declination = np.arcsin(np.sin(obliq) * np.sin(plon));

    return right_ascension, declination
# %%
