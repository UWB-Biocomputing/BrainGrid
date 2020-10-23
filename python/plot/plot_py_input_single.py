#!/Users/fumik/anaconda3/bin/python

#########################################################################
#
# @file         plot_py_input_single.py
# @author       Fumitaka Kawasaki
# @date         10/23/2020
#
# @brief        A python script to show the results of py_input_poisson_single.py
#               python script.
#
#########################################################################

import h5py
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import os

filename1 = '../results/single_historyDump.h5'
filename2 = '../results/vmLog.h5'

f1 = h5py.File(filename1, 'r')

#-------------------------------------------------------------------------
# Retrieve data from the HDF5 file
#-------------------------------------------------------------------------
spikesProbedNeurons = f1['spikesProbedNeurons'][()]      # spike events time of every neuron

basename1 = os.path.splitext(os.path.basename(filename1))[0]  # base name of the input file
[m, n] = spikesProbedNeurons.shape   # m - max number of event, n - number of neurons
firings_neuron = list()
firings_time = list()

for i in range(0, n):
    for j in range(0, m):
        if spikesProbedNeurons[j, i] != 0:
            firings_time.append(spikesProbedNeurons[j, i])
            firings_neuron.append(i)

# Read the membrane voltage record
basename2 = os.path.splitext(os.path.basename(filename2))[0]  # base name of the input file
f2 = h5py.File(filename2)
logTime = f2['logTime'][()]
logVm = f2['logVm'][()]

# Read the psr record
logPsr = f2['logPsr'][()]

#-------------------------------------------------------------------------
# fig_1: Plot spike raster graph
#-------------------------------------------------------------------------
fig_1 = plt.figure(figsize=(30,30))
ax_1 = fig_1.add_subplot(3,1,3)
ax_1.set_title('Spike raster graph: ' + basename1)
ax_1.set_xlabel('time (0.1 msec)')
ax_1.set_ylabel('neuron number')
ax_1.set_xlim([0,10000])
ax_1.set_ylim([0.5,1.5])
ax_1.eventplot(firings_time)

#-------------------------------------------------------------------------
# fig_2: Plot membrane voltage
#-------------------------------------------------------------------------
ax_2 = fig_1.add_subplot(3,1,2)
ax_2.set_title('Membrane voltage: ' + basename2)
ax_2.set_ylabel('membrane voltage (V)')
ax_2.set_xlim([0,10000])
ax_2.plot(logTime, logVm)

#-------------------------------------------------------------------------
# fig_3: Plot psr
#-------------------------------------------------------------------------
ax_3 = fig_1.add_subplot(3,1,1)
ax_3.set_title('psr: ' + basename2)
ax_3.set_ylabel('psr')
ax_3.set_xlim([0,10000])
ax_3.plot(logTime, logPsr)

plt.show()
