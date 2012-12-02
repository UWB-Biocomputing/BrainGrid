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

#include "LifNeuron.h"
#include "DynamicSpikingSynapse.h"

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
        pSummationMap(NULL)		
    {
    }

	//! Width of neuron map (assumes square)
	int width;

	//! Height of neuron map
	int height;

	//! Count of neurons in the simulation
	int cNeurons;

	//! Current simulation step
	int currentStep;

	//! Maximum number of simulation steps
	int maxSteps;

	//! The length of each step in simulation time
	FLOAT stepDuration;

	//! Maximum firing rate (only used by GPU simulation)
	int maxFiringRate;

	//! Maximum number of synapses per neuron (only used by GPU simulation)
	int maxSynapsesPerNeuron;

	//! Time elapsed between the beginning and end of the simulation step
	FLOAT deltaT;

	//! List of neurons
	vector<LifNeuron>* pNeuronList;

	//! The neuron type map (INH, EXC).
	neuronType* rgNeuronTypeMap;

	//! The starter existence map (T/F).
	bool* rgEndogenouslyActiveNeuronMap;

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

	//! List of lists of synapses (3d array)
	vector<ISynapse*>* rgSynapseMap;

	//! List of summation points
	FLOAT* pSummationMap;
};

#endif // _SIMULATIONINFO_H_
