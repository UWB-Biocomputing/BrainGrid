#!/Users/fumik/anaconda3/bin/python

#########################################################################
#
# @file		batch_h5.py
# @author	Fumitaka Kaewasaki
# @date		5/14/2020
#
# @brief	Main driver for plot figures
#
#########################################################################

import h5py
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import plotRadiusAndFiringRateChanges

args = sys.argv
filename = args[1]

f = h5py.File(filename, 'r')

#-------------------------------------------------------------------------
# Retrieve data from the HDF5 file
#-------------------------------------------------------------------------
Tsim = f['Tsim'][()]					# duration of epoch (s)
burstinessHist = f['burstinessHist'][()]		# burstiness history (every second)
neuronThresh = f['neuronThresh'][()]			# neuron's threshold of every neuron
neuronTypes = f['neuronTypes'][()]			# neurons's type of every neuron
radiiHistory = f['radiiHistory'][()]			# radii history of every neuron of every epoch
ratesHistory = f['ratesHistory'][()]			# rate history of every neuron of every epoch
simulationEndTime = f['simulationEndTime'][()]		# simulation end time (s)
spikesHistory = f['spikesHistory'][()]			# spike count of every neuron of every epoch
starterNeurons = f['starterNeurons'][()]		# starter neuron's indexes
xloc = f['xloc'][()]					# every neuron's x location
yloc = f['yloc'][()]					# every neuron's y location

#-------------------------------------------------------------------------
# Plot figures
#-------------------------------------------------------------------------

#
# Plot radius and firing rate changes and save the figures
#
plotRadiusAndFiringRateChanges.plot(filename, Tsim, neuronTypes, radiiHistory, ratesHistory, starterNeurons, xloc, yloc)

f.close()
