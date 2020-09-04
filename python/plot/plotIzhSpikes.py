#!/Users/fumik/anaconda3/bin/python

#########################################################################
#
# @file         plotIzhSpikes .py
# @author       Fumitaka Kawasaki
# @date         9/2/2020
#
# @brief        A python script to show the results of static_izh_1000.xml 
#               description file.
#
#########################################################################

import h5py
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import os

args = sys.argv
filename = args[1]

f = h5py.File(filename, 'r')

#-------------------------------------------------------------------------
# Retrieve data from the HDF5 file
#-------------------------------------------------------------------------
spikesProbedNeurons = f['spikesProbedNeurons'][()]      # spike events time of every neuron
spikesHistory = f['spikesHistory'][()]                  # spike count of every neuron of every epoch

basename = os.path.splitext(os.path.basename(filename))[0]  # base name of the input file
[m, n] = spikesProbedNeurons.shape   # m - max number of event, n - number of neurons
firings_neuron = list()
firings_time = list()

for i in range(0, n):
    for j in range(0, m):
        if spikesProbedNeurons[j, i] != 0:
            firings_time.append(spikesProbedNeurons[j, i])
            firings_neuron.append(i)

#-------------------------------------------------------------------------
# fig_1: Plot spike raster graph
#-------------------------------------------------------------------------
fig_1 = plt.figure(figsize=(20,12))
plt.title('Spike raster graph: ' + filename)
plt.xlabel('time (10 msec)')
plt.ylabel('neuron number')
plt.xlim([0,10000])
plt.ylim([0,1000])
plt.plot(firings_time, firings_neuron, '.')

#-------------------------------------------------------------------------
# fig_2: Plot populational spike activity
#-------------------------------------------------------------------------
fig_2 = plt.figure(figsize=(20,12))
plt.title('Populational spike activity: ' + filename)
plt.xlabel('time (10 msec)')
plt.ylabel('number of spikes')
plt.plot(spikesHistory)

plt.show()
