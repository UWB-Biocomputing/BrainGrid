#########################################################################
#
# @file         plotFinalRates.py
# @author       Fumitaka Kawasaki
# @date         5/27/2020
#
# @brief        Plot circles showing firing rate. 
#               lots circles at the unit locations (x, y) of radii propotional 
#               to firing rate, colored according to the neuron types for each 
#               neuron idx. It retunrs the handle graphics group corresponding 
#               to the circles.
#
#               Blue    Starter neurons
#               Red     Inhibitory neurons
#               Green   Excitatory neurons
#
#########################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
import os

def plot(filename, xloc, yloc, ratesHistory, neuronTypes, starterNeurons):
    INH = 1                             # type of inhobotory neurons
    EXC = 2                             # type of excitatory neurons
    basename = os.path.splitext(os.path.basename(filename))[0]  # base name of the input file
    
    [m, n] = ratesHistory.shape         # m - number of epoch, n - number of neurons
    finalRates = ratesHistory[m-1,:]    # total number of neurons
    maxRate = max(finalRates)
    if maxRate == 0:
        maxRate = 1
    numNeurons = neuronTypes.size
    bStarterNeurons = np.zeros(numNeurons, dtype=bool)
    bStarterNeurons[starterNeurons] = True # True if bStarterNeurons[i] is a starter neuron
   
    fig_1 = plt.figure(figsize=(20,20))
    ax = plt.axes()
    plt.title('Final Rates: ' + filename)
    
    for i in range(0, xloc.size):
        if neuronTypes[i] == EXC:
            if bStarterNeurons[i]:
                style = 'b'
            else:
                style = 'g'
        else:
            style = 'r'
        c = pat.Circle(xy=(xloc[i], yloc[i]), radius=0.5*finalRates[i]/maxRate, fc=style)
        
        ax.add_patch(c)
        
    plt.axis('scaled')
    fig_1.savefig(basename + '_FinalRates.png')
    plt.show()
    
    return
