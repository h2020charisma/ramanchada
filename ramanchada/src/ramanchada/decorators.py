# Written by Bastian Barton, Fraunhofer LBF, as part of the CHARISMA Work Package 4
# Version 0.2 (Prototype), May 5, 2021
# All rights reserved. The code may be used for purposes of CHARISMA as defined in the Consortium Agreement.

# MIT License
#
# Copyright (c) 2021–2022 CHARISMA H2020 project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import matplotlib.style as plot_style
import time
from functools import wraps


def specstyle(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        plot_style.use('bmh')
        plt.figure(figsize=[8,4])
        func(self, *args, **kwargs)
        plt.grid(None)
        plt.grid(axis='x', which='both', linestyle=':')
        # Fill if single plot
        lines = plt.gca().lines
        if len(lines) <= 1:
            for line in lines:
                x, y = line.get_xdata(), line.get_ydata()
                plt.fill_between(x, y, y.min(), alpha=0.2)
        # Show legend if not too many labels
        n_labels = len(plt.gca().get_legend_handles_labels()[0])
        if n_labels > 0 and n_labels < 10:
            plt.legend()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()
    return wrapper

def mark_peaks(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        for peak_x_pos, peak_y_pos in zip(self.bands['position'], self.bands['intensity']):
            peak_coords = (peak_x_pos, peak_y_pos + self.y.max()*.01)
            text_coords = (peak_x_pos, peak_y_pos + self.y.max()*.1)
            plt.gca().annotate(
                f'{peak_x_pos:.0f}',
                xy = peak_coords, xycoords = 'data',
                xytext = text_coords, textcoords='data',
                rotation=90,
                arrowprops = dict(facecolor='black', shrink=0.05, width=2, headwidth=5),
                horizontalalignment = 'center', verticalalignment='bottom', size=10
                )
    return wrapper

def log(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'log'):
            self.log.append([time.ctime(), func.__name__, args, kwargs])
        else:
            pass
        return func(self, *args, **kwargs)
    return wrapper

def change_y(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.y_label = "intensity [a.u.]"
        return func(self, *args, **kwargs)
    return wrapper

def change_x(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.x_label.endswith('modified'):
            self.x_label += " - modified"
        return func(self, *args, **kwargs)
    return wrapper