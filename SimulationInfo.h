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

#include "INeuron.h"
#include "DynamicSpikingSynapse.h"

//! Structure design to hold all of the parameters of the simulation.
struct SimulationInfo
{
    SimulationInfo() :
        width(0),
        height(0),
        cNeurons(0),
        currentStep(0),
        maxSteps(0),
        stepDuration(0),
        deltaT(0),
        pNeuronList(NULL),
        epsilon(0),
        beta(0),
        rho(0),
        maxRate(0),
        minRadius(0),
        startRadius(0),
        rgSynapseMap(NULL),
        pSummationMap(NULL),
		seed(0)
    {
    }

	void reset(int neurons, vector<INeuron*>* neronList, vector<ISynapse*>* synapseList, double* sumMap, FLOAT delta) {
		cNeurons = neurons;
		pNeuronList = neronList;
		rgSynapseMap = synapseList;
		pSummationMap = sumMap;
		deltaT = delta;
	}

	//! Width of neuron map (assumes square)
	int width;

	//! Height of neuron map
	int height;

	//! Count of neurons in the simulation
	int cNeurons;

	//! Current simulation step
	int currentStep;

	//! Maximum number of simulation steps NOTE: Not Currently Used
	int maxSteps;

	//! The length of each step in simulation time
	FLOAT stepDuration;

// NETWORK MODEL VARIABLES NMV-BEGIN {
	//! Maximum firing rate (only used by GPU simulation)
	int maxFiringRate;

	//! Maximum number of synapses per neuron (only used by GPU simulation)
	int maxSynapsesPerNeuron;
// } NMV-END

	//! Time elapsed between the beginning and end of the simulation step
	FLOAT deltaT;

	//! List of neurons
	vector<INeuron*>* pNeuronList;

	//! The neuron type map (INH, EXC).
	neuronType* rgNeuronTypeMap;

	//! The starter existence map (T/F).
	bool* rgEndogenouslyActiveNeuronMap;

// NETWORK MODEL VARIABLES NMV-BEGIN {
	//! growth param TODO: more detail here
	FLOAT epsilon;

	//! growth param TODO: more detail here
	FLOAT beta;

	//! growth param: change in radius scalar
	FLOAT rho;

	//! growth variable (m_targetRate / m_epsilon) TODO: more detail here
	FLOAT maxRate;

	//! The minimum possible radius.  We use this to prevent neurons from disconnecting from the network.
	FLOAT minRadius;

	//! The starting connectivity radius for all neurons.
	FLOAT startRadius;
// } NMV-END

	//! List of lists of synapses (3d array)
	vector<ISynapse*>* rgSynapseMap;

	//! List of summation points
	FLOAT* pSummationMap;

	//! Seed used for the simulation random SINGLE THREADED
	long seed;
};

#endif // _SIMULATIONINFO_H_
