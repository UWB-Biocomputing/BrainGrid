#########################################################################
#
# @file		plotRadiusAndFiringRateChanges.py
# @author	Fumitaka Kawasaki
# @date		5/14/2020
# 
# @brief	Plot radius and firing rate changes and save the figures
# 	
# 		Blue	Starter neurons
# 		Black	Exciatory neurons
# 		Red	Inhibitory neurons
# 		Green	Edge neurons
# 		Cyan	Corner neurons	
# 
#########################################################################

import h5py
import matplotlib.pyplot as plt
import math
import sys
import numpy as np
import os

def plot(filename, Tsim, neuronTypes, radiiHistory, ratesHistory, starterNeurons, xloc, yloc):
    INH = 1				# type of inhobotory neurons
    EXC = 2				# type of excitatory neurons
    numNeurons = neuronTypes.size	# total number of neurons
    xlen = math.sqrt(numNeurons)	# x length of the network
    ylen = xlen				# y length of the network
    basename = os.path.splitext(os.path.basename(filename))[0]	# base name of the input file

    #-------------------------------------------------------------------------
    # Make boolean index matrices for each neuron type
    #-------------------------------------------------------------------------
    bStarterNeurons = np.zeros(numNeurons, dtype=bool)
    bStarterNeurons[starterNeurons] = True
    inhNeurons = (neuronTypes == INH)    
    xedge = (xloc == 0) | (xloc == xlen-1)
    yedge = (yloc == 0) | (yloc == ylen-1)
    cornerNeurons = (xedge & yedge)
    edgeNeurons = (xedge | yedge) & ~(cornerNeurons | bStarterNeurons)
    excNeurons = (neuronTypes == EXC) & ~(cornerNeurons | bStarterNeurons | edgeNeurons)
    
    maxRadii = np.amax(radiiHistory)
    maxRates = np.max(ratesHistory)
    
    #-------------------------------------------------------------------------
    # fig_1_1: Plot radii and firing changes (combined)
    #-------------------------------------------------------------------------
    fig_1_1 = plt.figure(figsize=(20,12))
    ax_1_1 = fig_1_1.add_subplot(2,1,1)    
    ax_1_1.set_title('Radius plot: ' + filename)
    ax_1_1.set_xlabel('Time (s) × %d' % Tsim)
    ax_1_1.set_ylabel('Radii')    
    ax_1_1.set_ylim([0,maxRadii])
    
    p1 = ax_1_1.plot(radiiHistory[:,bStarterNeurons], 'b')
    p2 = ax_1_1.plot(radiiHistory[:,excNeurons], 'k')   
    p3 = ax_1_1.plot(radiiHistory[:,inhNeurons], 'r')
    p4 = ax_1_1.plot(radiiHistory[:,edgeNeurons], 'g')
    p5 = ax_1_1.plot(radiiHistory[:,cornerNeurons], 'c')
       
    ax_1_2 = fig_1_1.add_subplot(2,1,2)
    ax_1_2.set_title('Firing rate plot: ' + filename)
    ax_1_2.set_xlabel('Time (s) × %d' % Tsim)
    ax_1_2.set_ylabel('Firing rate (Hz)')
    ax_1_2.set_ylim([0,maxRates])
    
    ax_1_2.plot(ratesHistory[:,bStarterNeurons], 'b')
    ax_1_2.plot(ratesHistory[:,excNeurons], 'k')   
    ax_1_2.plot(ratesHistory[:,inhNeurons], 'r')
    ax_1_2.plot(ratesHistory[:,edgeNeurons], 'g')
    ax_1_2.plot(ratesHistory[:,cornerNeurons], 'c')  
    
    labels = ['starter neurons', 'excitatory neurons', 'inhibitory neurons', 'edge neurons', 'corner neurons']
    fig_1_1.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), labels, loc='upper right')
    fig_1_1.savefig(basename + '_RFRC_fig_1_1.png')
 
    #-------------------------------------------------------------------------
    # fig_1_2: Plot radii and firing changes (starter neurons)
    #-------------------------------------------------------------------------
    fig_1_2 = plt.figure(figsize=(20,12))
    ax_2_1 = fig_1_2.add_subplot(2,1,1)    
    ax_2_1.set_title('Radius plot (starter neurons): ' + filename)
    ax_2_1.set_xlabel('Time (s) × %d' % Tsim)
    ax_2_1.set_ylabel('Radii')  
    ax_2_1.set_ylim([0,maxRadii])
    ax_2_1.plot(radiiHistory[:,bStarterNeurons], 'b')
    
    ax_2_2 = fig_1_2.add_subplot(2,1,2)
    ax_2_2.set_title('Firing rate plot (starter neurons): ' + filename)
    ax_2_2.set_xlabel('Time (s) × %d' % Tsim)
    ax_2_2.set_ylabel('Firing rate (Hz)')
    ax_2_2.set_ylim([0,maxRates])
    ax_2_2.plot(ratesHistory[:,bStarterNeurons], 'b')
    
    fig_1_2.savefig(basename + '_RFRC_fig_1_2.png')
    
    #-------------------------------------------------------------------------
    # fig_1_3: Plot radii and firing changes (excitatory neurons)
    #-------------------------------------------------------------------------
    fig_1_3 = plt.figure(figsize=(20,12))
    ax_3_1 = fig_1_3.add_subplot(2,1,1)    
    ax_3_1.set_title('Radius plot (excitatory neurons): ' + filename)
    ax_3_1.set_xlabel('Time (s) × %d' % Tsim)
    ax_3_1.set_ylabel('Radii')   
    ax_3_1.set_ylim([0,maxRadii])
    ax_3_1.plot(radiiHistory[:,excNeurons], 'k')
    
    ax_3_2 = fig_1_3.add_subplot(2,1,2)
    ax_3_2.set_title('Firing rate plot (excitatory neurons): ' + filename)
    ax_3_2.set_xlabel('Time (s) × %d' % Tsim)
    ax_3_2.set_ylabel('Firing rate (Hz)')
    ax_3_2.set_ylim([0,maxRates])
    ax_3_2.plot(ratesHistory[:,excNeurons], 'k') 
    
    fig_1_3.savefig(basename + '_RFRC_fig_1_3.png')
    
    #-------------------------------------------------------------------------
    # fig_1_4: Plot radii and firing changes (inhibitory neurons)
    #-------------------------------------------------------------------------
    fig_1_4 = plt.figure(figsize=(20,12))
    ax_4_1 = fig_1_4.add_subplot(2,1,1)    
    ax_4_1.set_title('Radius plot (inhibitory neurons): ' + filename)
    ax_4_1.set_xlabel('Time (s) × %d' % Tsim)
    ax_4_1.set_ylabel('Radii') 
    ax_4_1.set_ylim([0,maxRadii])
    ax_4_1.plot(radiiHistory[:,inhNeurons], 'r')
    
    ax_4_2 = fig_1_4.add_subplot(2,1,2)
    ax_4_2.set_title('Firing rate plot (inhibitory neurons): ' + filename)
    ax_4_2.set_xlabel('Time (s) × %d' % Tsim)
    ax_4_2.set_ylabel('Firing rate (Hz)')
    ax_4_2.set_ylim([0,maxRates])
    ax_4_2.plot(ratesHistory[:,inhNeurons], 'r')
    
    fig_1_4.savefig(basename + '_RFRC_fig_1_4.png')
    
    #-------------------------------------------------------------------------
    # fig_1_5: Plot radii and firing changes (edge neurons)
    #-------------------------------------------------------------------------
    fig_1_5 = plt.figure(figsize=(20,12))
    ax_5_1 = fig_1_5.add_subplot(2,1,1)    
    ax_5_1.set_title('Radius plot (edge neurons): ' + filename)
    ax_5_1.set_xlabel('Time (s) × %d' % Tsim)
    ax_5_1.set_ylabel('Radii') 
    ax_5_1.set_ylim([0,maxRadii])
    ax_5_1.plot(radiiHistory[:,edgeNeurons], 'g')
    
    ax_5_2 = fig_1_5.add_subplot(2,1,2)
    ax_5_2.set_title('Firing rate plot (edge neurons): ' + filename)
    ax_5_2.set_xlabel('Time (s) × %d' % Tsim)
    ax_5_2.set_ylabel('Firing rate (Hz)')
    ax_5_2.set_ylim([0,maxRates])
    ax_5_2.plot(ratesHistory[:,edgeNeurons], 'g')
    
    fig_1_5.savefig(basename + '_RFRC_fig_1_5.png')
    
    #-------------------------------------------------------------------------
    # fig_1_6: Plot radii and firing changes (corner neurons)
    #-------------------------------------------------------------------------
    fig_1_6 = plt.figure(figsize=(20,12))
    ax_6_1 = fig_1_6.add_subplot(2,1,1)    
    ax_6_1.set_title('Radius plot (corner neurons): ' + filename)
    ax_6_1.set_xlabel('Time (s) × %d' % Tsim)
    ax_6_1.set_ylabel('Radii') 
    ax_6_1.set_ylim([0,maxRadii])
    ax_6_1.plot(radiiHistory[:,cornerNeurons], 'c')
    
    ax_6_2 = fig_1_6.add_subplot(2,1,2)
    ax_6_2.set_title('Firing rate plot (corner neurons): ' + filename)
    ax_6_2.set_xlabel('Time (s) × %d' % Tsim)
    ax_6_2.set_ylabel('Firing rate (Hz)')
    ax_6_2.set_ylim([0,maxRates])
    ax_6_2.plot(ratesHistory[:,cornerNeurons], 'c')
    
    fig_1_6.savefig(basename + '_RFRC_fig_1_6.png')

    return
