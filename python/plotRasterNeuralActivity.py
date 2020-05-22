#########################################################################
#
# @file         plotRasterNeuralActivity.py
# @author       Fumitaka Kawasaki
# @date         5/22/2020
#
# @brief        Creates a raster plot of neural activity. A point is plotted for
#               each neuron that has a non-zero firing rate. Y axis is neuron
#               number; X axis is time.
#
#               filename    Input file name
#               rates       Firing rate history of neurons
#               Tsim        Duration of one epoch (s)
#
#########################################################################

import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import os

def plot(filename, rates, Tsim):
    basename = os.path.splitext(os.path.basename(filename))[0]  # base name of the input file
    [m, n] = rates.shape   # m - number of epoch, n - number of neurons
    
    fig_1 = plt.figure(figsize=(20,12))
    plt.title('Raster plot: ' + filename)
    plt.xlabel('Time (s) x %d' % Tsim)
    plt.ylabel('Neuron Number')
    plt.xlim([0,m])
    plt.ylim([0,n])
    
    for c in range(0, n):
        st = [i for i, x in enumerate(rates[:, c]) if x != 0]
        plt.plot(st, c*np.ones(len(st)), 'k')
    
    fig_1.savefig(basename + '_RPNA_fig_1.png')
    plt.show()
    
    return
