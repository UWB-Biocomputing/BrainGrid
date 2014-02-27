/**
 *      @file SimulationInfo.h
 *
 *      @brief Header file for SimulationInfo.
 */
//! Simulation information.

/**
 ** \class SimulationInfo SimulationInfo.h "SimulationInfo.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The SimulationInfo contains all information necessary for the simulation.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Allan Ortiz & Cory Mayberry
 **/

#pragma once

#ifndef _SIMULATIONINFO_H_
#define _SIMULATIONINFO_H_


//! Structure design to hold all of the parameters of the simulation.
struct SimulationInfo
{
    SimulationInfo() :
        width(0),
        height(0),
        totalNeurons(0),
        currentStep(0),
        maxSteps(0),
        epochDuration(0),
        deltaT(0),
        //pNeuronList(NULL), // NOT NEEDED?
        epsilon(0),
        beta(0),
        rho(0),
        maxRate(0),
        minRadius(0),
        startRadius(0),
        //rgSynapseMap(NULL), // NOT NEEDED?
        pSummationMap(NULL),
		seed(0)
    {
    }

/* NOT NEEDED?
	void reset(int neurons, vector<INeuron*>* neronList, vector<ISynapse*>* synapseList, BGFLOAT* sumMap, BGFLOAT delta) {
		cNeurons = neurons;
		pNeuronList = neronList;
		rgSynapseMap = synapseList;
		pSummationMap = sumMap;
		deltaT = delta;
	}
*/
	//! Width of neuron map (assumes square)
	int width;

	//! Height of neuron map
	int height;

	//! Count of neurons in the simulation
	int cNeurons;

	//! Current simulation step
	int currentStep;

	//! Maximum number of simulation steps
	// Not clear what this is, but it is being used by the Simulator.writeMemory methods
	// Those methods are not currently functional.
	int maxSteps; // TODO: delete

	//! The length of each step in simulation time
	BGFLOAT epochDuration; // Epoch duration !!!!!!!!

// NETWORK MODEL VARIABLES NMV-BEGIN {
	//! Maximum firing rate. **Only used by GPU simulation.**
	int maxFiringRate;

	//! Maximum number of synapses per neuron. **Only used by GPU simulation.**
	int maxSynapsesPerNeuron;
// } NMV-END

	//! Time elapsed between the beginning and end of the simulation step
	BGFLOAT deltaT; // Inner Simulation Step Duration !!!!!!!!

	//! List of neurons
	//vector<INeuron*>* pNeuronList; // NOT NEEDED?

	//! The neuron type map (INH, EXC).
	neuronType* rgNeuronTypeMap;

	//! The starter existence map (T/F).
	bool* rgEndogenouslyActiveNeuronMap;

// NETWORK MODEL VARIABLES NMV-BEGIN {
	//! growth param TODO: more detail here
	BGFLOAT epsilon;

	//! growth param TODO: more detail here
	BGFLOAT beta;

	//! growth param: change in radius scalar
	BGFLOAT rho;

	//! growth variable (m_targetRate / m_epsilon) TODO: more detail here
	BGFLOAT maxRate;

	//! The minimum possible radius.  We use this to prevent neurons from disconnecting from the network.
	BGFLOAT minRadius;

	//! The starting connectivity radius for all neurons.
	BGFLOAT startRadius;
// } NMV-END

	//! List of lists of synapses (3d array)
	//vector<ISynapse*>* rgSynapseMap; // NOT NEEDED?

	//! List of summation points
	BGFLOAT* pSummationMap;

	//! Seed used for the simulation random SINGLE THREADED
	long seed;
};

#endif // _SIMULATIONINFO_H_
