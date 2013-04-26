/**
 ** \brief A class that performs the simulation on GPU.
 **
 ** \class GpuSim GpuSim.h "GpuSim.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The GpuSim performs updating neurons and synapses of one activity epoch, and
 ** thereafter updating network using kernel functions on GPU. 
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 ** 
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

/**
 ** \file GpuSim.h
 **
 ** \brief Header file for GpuSim.
 **/

#pragma once

#ifndef _GPUSIM_H_
#define _GPUSIM_H_

#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "ISimulation.h"
#include "DynamicSpikingSynapse_struct.h"
#include "LifNeuron_struct.h"
#include "HostSim.h"
#include "global.h"


class GpuSim : public ISimulation
{
public:
    //! The constructor for GpuSim.
    GpuSim(SimulationInfo* psi);
    ~GpuSim();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, VectorMatrix& xloc, VectorMatrix& yloc);

    //! Terminate process.
    virtual void term(SimulationInfo* psi);

    //! Initialize radii
    virtual void initRadii(VectorMatrix& newRadii);

    //! Performs updating neurons and synapses for one activity epoch.
    virtual void advanceUntilGrowth(SimulationInfo* psi);

    //! Updates the network.
    virtual void updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory);

	//! Returns a type of Neuron to be used in the Network
	virtual INeuron* returnNeuron();

#ifdef STORE_SPIKEHISTORY
    //! pointer to an array to keep spike history for one activity epoch
    uint64_t* spikeArray;
#endif // STORE_SPIKEHISTORY

    //! list of spike count for each neuron
    int* spikeCounts;

private:
    void printComparison(LifNeuron_struct* neuron_st, vector<LifNeuron*>* neuronObjects, int i);
	
    //! Copy synapse and neuron C++ objects into C structs.
    void dataToCStructs( SimulationInfo* psi, LifNeuron_struct* neuron_st, DynamicSpikingSynapse_struct* synapse_st ); 

    //! Adds a synapse to the network.  Requires the locations of the source and destination neurons.
    DynamicSpikingSynapse& addSynapse(SimulationInfo* psi, int source_x, int source_y, int dest_x, int dest_y);

    //! Returns the type of synapse at the given coordinates.
    synapseType synType(SimulationInfo* psi, Coordinate a, Coordinate b);

    //! Return 1 if originating neuron is excitatory, -1 otherwise.
    int synSign(synapseType t);

    //! Print network radii to console.
    void printNetworkRadii(SimulationInfo* psi, VectorMatrix networkRadii) const;

    //! synapse weight
    CompleteMatrix W;

    //! neuron radii
    VectorMatrix radii;

    //! spiking rate
    VectorMatrix rates;

    //! Inter-neuron distance squared    CompleteMatrix dist2;
    CompleteMatrix dist2;

    //! distance between connection frontiers
    CompleteMatrix delta;

    //! the true inter-neuron distance
    CompleteMatrix dist;

    //! areas of overlap
    CompleteMatrix area;

    //! neuron's outgrowth
    VectorMatrix outgrowth;

    //! displacement of neuron radii
    VectorMatrix deltaR;
};

#endif // _GPUSIM_H_
