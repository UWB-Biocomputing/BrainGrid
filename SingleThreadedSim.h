/**
 *      @file SingleThreadedSim.h
 *
 *      @brief Header file for SingleThreadedSim.
 */
//! A class that performs the single threaded simulation on CPU.

/**
 ** \class SingleThreadedSim SingleThreadedSim.h "SingleThreadedSim.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The SingleThreadedSim performs updating neurons and synapses of one activity epoch, and
 ** thereafter updating network using single threaded functions on CPU. 
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

#ifndef _SINGLETHREADEDSIM_H_
#define _SINGLETHREADEDSIM_H_

#include "global.h"
#include "ISimulation.h"
#include "HostSim.h"

#define IOCP_KEY_NEURON 0
#define IOCP_KEY_SYNAPSE 1

class SingleThreadedSim : public HostSim
{
public:
    //! The constructor for SingleThreadedSim.
    SingleThreadedSim(SimulationInfo* psi);
    ~SingleThreadedSim();

    //! Perform updating neurons and synapses for one activity epoch.
    virtual void advanceUntilGrowth(SimulationInfo* psi);

    //! Update the network.
    virtual void updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory);

	void worker1();

private:
    //! Perform updating neurons for one time step.
    void advanceNeurons(SimulationInfo* psi);

    //! Perform updating synapses for one time step.
    void advanceSynapses(SimulationInfo* psi);

	SimulationInfo* m_psi;
	HANDLE m_EventAdvanceNeurons;
	HANDLE m_EventAdvanceNeuronsComplete;
	int m_I; // Out current "index" for advance stages
	uint64_t m_Count;
	uint64_t m_EndStep;
	int m_MaxThreads;
	int m_StepsPerIteration;
	volatile LONG m_OpsCompleted;
	HANDLE m_hIOCP;
};

#endif // _SINGLETHREADEDSIM_H_
