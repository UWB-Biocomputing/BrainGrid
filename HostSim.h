/**
 *	@file HostSim.h
 *
 *	@brief Header file for HostSim.
 */
//! A super class of MultiThreadedSim and SingleThreadedSim classes.

/**
 ** \class HostSim HostSim.h "HostSim.h"
 **
 ** \latexonly	\subsubsection*{Implementation} \endlatexonly
 ** \htmlonly	<h3>Implementation</h3> \endhtmlonly
 **
 ** The HostSim provides a common functions and data structure for simulations on host computer.
 **
 ** \latexonly	\subsubsection*{Credits} \endlatexonly
 ** \htmlonly	<h3>Credits</h3> \endhtmlonly
 ** 
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **	@author Fumitaka Kawasaki
 **/

#pragma once

#ifndef _HOSTSIM_H_
#define _HOSTSIM_H_

#include "ISimulation.h"
#include "LifNeuron.h"

class HostSim : public ISimulation
{
public:
    //! The constructor for HostSim.
    HostSim(SimulationInfo* psi);
    ~HostSim();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, VectorMatrix& xloc, VectorMatrix& yloc);

    //! Terminate process.
    virtual void term(SimulationInfo* psi);
  
    //! Initialize radii
    virtual void initRadii(VectorMatrix& newRadii);

	//! Returns a type of Neuron to be used in the Network HOW IS THIS USED?
	virtual INeuron* returnNeuron();

protected:
// NETWORK MODEL VARIABLES NMV-BEGIN {
    //! Adds a synapse to the network.  Requires the locations of the source and destination neurons.
    ISynapse* addSynapse(SimulationInfo* psi, int source_x, int source_y, int dest_x, int dest_y);
// } NMV-END

    // ARE THE FOLLOWING METHODS USEFUL

    //! Returns the type of synapse at the given coordinates.
    synapseType synType(SimulationInfo* psi, Coordinate a, Coordinate b);

    //! Return 1 if originating neuron is excitatory, -1 otherwise.
    int synSign(synapseType t);

    //! Print network radii to console.
    void printNetworkRadii(SimulationInfo* psi, VectorMatrix networkRadii) const;

    // -------------------------------------

// NETWORK MODEL VARIABLES NMV-BEGIN {
    //! synapse weight
    CompleteMatrix W;

    //! neuron radii
    VectorMatrix radii;

    //! spiking rate
    VectorMatrix rates;

    //! Inter-neuron distance squared
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
// } NMV-END
};

#endif // _HOSTSIM_H_
