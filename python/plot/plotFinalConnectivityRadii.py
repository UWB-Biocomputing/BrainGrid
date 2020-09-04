#########################################################################
#
# @file         plotFinalConnectivityRadii.py
# @author       Fumitaka Kawasaki
# @date         5/27/2020
#
# @brief        Plot circles showing connectivity radii. 
#               Plots points at the unit locations (x, y) and then plots 
#               circles of radii colored according to the neuron types 
#               for each neuron idx. 
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

def plot(filename, xloc, yloc, radiiHistory, neuronTypes, starterNeurons):
    INH = 1                             # type of inhobotory neurons
    EXC = 2                             # type of excitatory neurons
    basename = os.path.splitext(os.path.basename(filename))[0]  # base name of the input file
    
    [m, n] = radiiHistory.shape         # m - number of epoch, n - number of neurons
    finalRadii = radiiHistory[m-1,:]    # total number of neurons
    numNeurons = neuronTypes.size
    bStarterNeurons = np.zeros(numNeurons, dtype=bool)
    bStarterNeurons[starterNeurons] = True # True if bStarterNeurons[i] is a starter neuron
   
    fig_1 = plt.figure(figsize=(20,20))
    ax = plt.axes()
    plt.title('Final Radii: ' + filename)
    plt.plot(xloc, yloc, 'ok')          # plots points at the unit locations (x, y)
    

    for i in range(0, xloc.size):
        if neuronTypes[i] == EXC:
            if bStarterNeurons[i]:
                style = 'b'
            else:
                style = 'g'
        else:
            style = 'r'
        c = pat.Circle(xy=(xloc[i], yloc[i]), radius=finalRadii[i], ec=style, fill=False)
        ax.add_patch(c)
        
    plt.axis('scaled')
    fig_1.savefig(basename + '_ConnectivityRadii.png')
    
    return

