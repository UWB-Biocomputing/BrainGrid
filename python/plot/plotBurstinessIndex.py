#########################################################################
#
# @file         plotBurstinessIndex.py
# @author       Fumitaka Kawasaki
# @date         5/20/2020
#
# @brief        Plot burstiness index vs. time for spike records.
#
#               A burstiness index was computed by first clculating a spike
#               count vs. time histogram for the entire network during the
#               whole simulation period. The fraction, f15, of the total
#               number of spikes containing by the 15% most populations bins
#               was then normalized to produce the burstiness index, BI, as
#               BI = (f15 - 0.15) / 0.85.
#
#               filename    Input file name
#               hist        History of network wide spike count per second
#
#########################################################################

import matplotlib.pyplot as plt 
import math 
import sys 
import numpy as np
import os

def plot(filename, hist): 
    basename = os.path.splitext(os.path.basename(filename))[0]  # base name of the input file
    
    numbins = hist.size 
    seglength = 300                       # segment (window) length
    f15 = np.zeros(numbins-seglength+1)   # array to store f15 values
    
    print('\nComputing f15 at offset: ')
    
    #-------------------------------------------------------------------------
    # Make f15 matrix for each window
    #-------------------------------------------------------------------------
    for n in range(0, numbins-seglength+1):
        '''
        if n % 1000 == 0:
            print(n)
        if n % 10000 == 0:
            print('\n')
        '''
        bins = sorted(hist[n:n+seglength])     # bin values; smallest to largest
        spikes = sum(bins)                     # total number of spikes in window
        sp15 = sum(bins[round(0.85*seglength):seglength])  # spikes in largest 15%
        if (sp15 == 0) and (spikes == 0):
            f15[n] = 0
        else:
            f15[n] = sp15 / spikes
    
    #-------------------------------------------------------------------------
    # fig_1: Plot burstiness index vs. time for spike records
    #-------------------------------------------------------------------------
    fig_1 = plt.figure(figsize=(20,12))
    plt.title('Burstiness index: ' + filename)
    plt.xlabel('Time (s)')
    plt.ylabel('Burstiness Index')
    plt.plot((f15-0.15) / 0.85)
    fig_1.savefig(basename + '_BurstinessIndex.png')

    return


