# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

from matplotlib.lines import Line2D
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd

def wavelengths_to_wavenumbers(wavelenght_vector, laser_wavelength):
    return (1./laser_wavelength - 1./wavelenght_vector) * 1e7

def hqi(y1, y2):
    # Hit quality index (equivalent to cross-correlation)
    # See Rodriguez, J.D., et al., Standardization of Raman spectra for transfer of spectral libraries across different
    # instruments. Analyst, 2011. 136(20): p. 4232-4240.
    return np.linalg.norm(np.dot(y1, y2))**2 / np.linalg.norm(y1)**2 / np.linalg.norm(y2)**2

def lims(x, xmin, xmax):
    def l(y):
        return y[(x>xmin) & (x<xmax)]
    return l

#def lims(x, x_min, x_max):
#    x_min = np.fmax(x_min, np.fmin(x[0], x[-1]))
#    x_max = np.fmin(x_max, np.fmax(x[0], x[-1]))
#    #y_min, y_max = np.argmin(np.abs(x-x_min)), np.argmin(np.abs(x-x_max))
##    y_min, y_max = np.argmin(np.abs(x-x_max)),np.argmin(np.abs(x-x_min))
#    def l(Y):
#        return Y[...,y_min:y_max+1]
#    return l

def interpolation_within_bounds(cal_x, cal_y, poly_degree):
    coeffs = np.polyfit(cal_x, cal_y, poly_degree)
    def interp(x):
        shifts = np.poly1d(coeffs)(x)
        over, under = x > cal_x.min()-30, x < cal_x.max()+30
        if not np.any(over*under):
            return np.zeros_like(x)
        if not np.all(over):
            shifts[~over] = shifts[over][0]
        if not np.all(under):
            shifts[~under] = shifts[under][-1]
        return shifts
    return interp

def colorByTarget(*targets, cmap=plt.cm.coolwarm):
    lines = plt.gca().get_lines()
    if len(targets) == 0: return
    if len(targets[0]) != len(lines): return
    # get rid of old legends
    plt.gca().legend_ = None
    #plt.draw()
    # 1st target is color
    if all((isinstance(item, float) or isinstance(item, int)) for item in targets[0]):
    #if not any(isinstance(item, str) for item in targets[0]):
        # 1st target is all numeric
        colors, leg  = num2colors(targets[0], cmap=cmap)     
    else:
        colors, leg = cat2colors(targets[0])
    for line, color in zip(lines, colors):
        line.set_color(color)
    plt.gca().add_artist(leg)
    # 2nd target is style
    if len(targets) > 1:
        if len(targets[1]) != len(lines): return
        styles, leg2 = cat2sytles(targets[1])
        for line, st in zip(lines, styles):
            line.set_linestyle(st)
        plt.gca().add_artist(leg2)
    #plt.legend()
    return

def cat2sytles(targets):
    # put most abundant category first!
    categories = list(set(targets))
    categories = sorted(categories, key=lambda x: targets.count(x), reverse=True)
    styles = dict( zip( categories, cycle(['-', '--', ':', '-.']) ) )
    sty = [styles[t] for t in targets]
    custom_lines = [] 
    for c in categories:
        custom_lines.append(Line2D([0], [0], linestyle=styles[c], lw=2))
    leg = plt.legend(custom_lines, [str(s) for s in categories ], loc=2)
    return sty, leg

def cat2colors(targets):
    # put most abundant category first!
    categories = list(set(targets))
    categories = sorted(categories, key=lambda x: targets.count(x), reverse=True)
    #colors = dict( zip(categories, 'rbgcmk') )
    colcycle = ["C"+str(n) for n in range(len(categories))]
    colors = dict( zip(categories, colcycle) )
    col = [colors[t] for t in targets]
    custom_lines = []
    for s in categories:
        custom_lines.append(Line2D([0], [0], color=colors[s], lw=4))
    leg = plt.legend(custom_lines, [str(s) for s in categories], loc=1)
    return col, leg

def num2colors(targets, cmap=plt.cm.coolwarm):
    dd = np.array(targets)*1.
    d = dd.copy()
    d -= d.min()
    d /= d.max()
    colors = [cmap(t) for t in d]
    # Make legend
    custom_lines = []
    d = sorted([s for s in set(d)])
    dd = sorted([s for s in set(dd)])
    for s in d:
        custom_lines.append(Line2D([0], [0], color=cmap(s), lw=4))
    leg = plt.legend(custom_lines, [str(s) for s in dd], loc=1)
    return colors, leg

def labels_from_filenames(filenames, pivot_string=None, pos=0, numeric=True, length=False):
    l = []
    for f in filenames:
        if pivot_string != None:
            pivot_pos = re.search(pivot_string, f)
            if pivot_pos != None:
                pivot_span = pivot_pos.span()
                if pos < 0:
                    f = f[:pivot_span[0]]
                else:
                    f = f[pivot_span[1]:]
        if numeric:
            regexp = r"[0-9']+"
        else:
            regexp = r"[a-z']+"
        labels = re.findall(regexp, f, re.I)
        if length:
            labels = sorted(labels, key=len, reverse=True)
        if numeric:
            labels = [int(n) for n in labels]
        if labels == []:
            label = None
        elif pos > len(labels)-1:
            label = labels[-1]
        else:
            label = labels[pos]
        l.append(label)
    return l
