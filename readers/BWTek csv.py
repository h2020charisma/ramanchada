# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:21:47 2018

@author: bartonba
"""
from __future__ import print_function
import numpy as np
import csv
from tkFileDialog import askopenfilename
from Tkinter import Tk
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import wiener
from scipy import sparse
from scipy.sparse.linalg import spsolve

def readRamanDVS(filename=[]):
    if filename == []:
        Tk().withdraw()
        filename = askopenfilename()
        print(filename)
    d = pd.read_csv(filename, sep=';', header=None, nrows=1, skiprows=1)
    ts = pd.Timestamp(d[1][0])
    k = []
    y = []
    with open(filename,'rb') as f:
        c = csv.reader(f,delimiter=';')
        for ii in range(0,80):
            next(c)
        for row in c:
            k.append(np.double(row[3]))
            y.append(np.double(row[6]))
    k = np.array(k)
    y = np.array(y)
    return k, y, ts

def readRamanDVS_series(name, n_stop, n_start=1, n_digits=4, plot=False):
    # Start & stop numbers of opus files (n1 should be 0)
    numbers = [n_start, n_stop]
    # Allocate arrays for data, set counters
    S = np.zeros([numbers[1]-numbers[0]+1, 2048])
    jj = 0
    t = []
    # Load each spectrum into line of S
    for ii in range(numbers[0], numbers[1]+1):
        filename = name + ('{:0' +str(n_digits)+'.0f}').format(ii) + '.txt'
        [k, spec, ts] = readRamanDVS(filename)
        print('\rReading spectrum # ' + str(ii), end='\r')
        # Read spectrum & time
        S[jj,:] = spec
        if jj == 0:
            first_ts = ts
            t.append(0)
        else:
            delta = (ts - first_ts).total_seconds()/60.
            t.append(delta)
        jj += 1
    if plot:
        plt.figure()
        plt.plot(k, S[0,...], label='First')
        plt.plot(k, np.mean(S,0), label='Average')
        plt.plot(k, np.std(S,0), label='Standard dev.')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Intensity [a.u.]')
        plt.title('Raman spectrum #0 from series', fontsize=10)
        plt.grid(linestyle = ':')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels)
    return k, S, t

def baseline_als(spec, lam=5e4, p=0.001, niter=100, smooth=7, show=True):
    # After Eilers, P. H., & Boelens, H. F. (2005). Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre Report, 1(1), 5.
    y = spec.copy()
    if smooth > 0: y = wiener(y, smooth)
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    if show:
        plt.figure()
        plt.plot(spec)
        plt.plot(z)
    return z

def stack_baseline_correct(stack, niter=10, show=True):#, norm=False):
    corrStack = stack.copy()
    for ii in  range(np.shape(stack)[0]):
        #print('\rReading spectrum # ' + str(ii), end='\r')
        print('\rbackground corr. of spectrum # ' + str(ii), end='\r')
        corrStack[ii,...] -= baseline_als(stack[ii,...], niter=niter, show=False)
        corrStack[ii,...] -= np.min(corrStack[ii,...])
        #if norm: corrStack[ii,...] /= np.max(corrStack[ii,...])
    if show:
        l = []
        plt.figure()
        plt.plot(np.mean(stack,0))
        l.append('average of raw spectra')
        plt.plot(np.mean(corrStack,0))
        l.append('average of corrected spectra')
        plt.legend(l)
    return corrStack

def stack_equalize(stack, func=np.max):
    corrStack = stack.copy()
    for ii in  range(np.shape(stack)[0]):
        corrStack[ii,...] -= np.min(corrStack[ii,...])
        corrStack[ii,...] /= func(corrStack[ii,...])
    return corrStack

def demixRamanDVS_series(k, S, t, n_components=2, k_start=-200, k_stop=5000, normalize_components=True, smooth=0):
    i_start = np.argmin(np.abs(k-k_start))
    i_stop = np.argmin(np.abs(k-k_stop))
    k = k[i_start:i_stop]
    S = S[...,i_start:i_stop]
    # Define NMF model, do fit 7 transform. H contains comp. spectra as lines, W contains concentrations as columns.
    model = NMF(n_components=n_components)
    W = model.fit_transform(S)
    H = model.components_
    if normalize_components:
        # Scale components & concentrations such that spectra have same mean (realistic / physical).
        # Note that S_n * c_n remains constant!
        for ii in range(H.shape[0]):
            factor = np.max(H[ii,...])
            H[ii,...] /= factor
            W[...,ii] *= factor
    # Plot component spectra
    plt.figure()
    d = 0
    l = []
    #m =  np.mean(S,0)
    #m /= np.mean(m)
    #plt.plot(k, m + d, 'k-')
    #d += np.max(m)
    #l.append('avg spectrum')
    for ii in range(H.shape[0]):
        plt.plot(k, H[ii,...] + d)
        d += np.max(H[ii,...])
        l.append('comp. #' + str(ii))
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Intensity [a.u.]')
    plt.yticks([])
    plt.title('Multivariate component spectra', fontsize=10)
    plt.legend(l)
    # plot concentrations
    plt.figure()
    l = []
    for ii in range(H.shape[0]):
        w = W[:,ii].copy()
        if smooth > 0: w = wiener(w, smooth)
        plt.plot(t, w)
        l.append('comp. #' + str(ii))
    plt.xlabel('time [min.]')
    plt.ylabel('realtive concentration [a.u.]')
    plt.title('Multivariate component concentrations', fontsize=10)
    plt.grid(linestyle = ':')
    plt.legend(l)
    return W, H, k

def rollingBall(spec, R, smooth=0):
    if smooth > 0: spec = wiener(spec, smooth)
    bg = spec.copy()
    n = np.arange(np.size(spec))
    spec2D = np.vstack((n,spec))
    # Für jeden Kanal
    for pos in n:
        # Ball aufsteigen lassen von Intervall-Minimum
        heights = np.linspace(np.min(spec)-R,np.max(spec),100)
        #h = np.min(spec)-R-1
        for h in heights:
            p = np.array([pos,h])
            p = p[:, np.newaxis]
            dist = np.linalg.norm(spec2D-p,axis=0)
            #print np.min(dist)
            # Sobald der erste Wert den Ball berührt, Höhe + Radius als Background setzten
            if any(dist < R):
                break
        bg[pos] = h+R
    plt.figure()
    plt.plot(spec)
    plt.plot(bg)
    return bg

def ca():
    plt.close('all')