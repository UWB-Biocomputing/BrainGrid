/**
 *      @file MultiThreadedSim.h
 *
 *      @brief Header file for MultiThreadedSim.h
 */
//! A class that performs the multi threaded simulation on CPU.

/**
 ** \class MultiThreadedSim MultiThreadedSim.h "MultiThreadedSim.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The MultiThreadedSim performs updating neurons and synapses of one activity epoch, and
 ** thereafter updating network using multi threaded functions on CPU. 
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

#ifndef _MULTITHREADEDSIM_H_
#define _MULTITHREADEDSIM_H_

#include "HostSim.h"
#include <omp.h>

class MultiThreadedSim : public HostSim
{
public:
    //! The constructor for MultiThreadedSim.
    MultiThreadedSim(SimulationInfo* psi);
    ~MultiThreadedSim();

    //! Perform updating neurons and synapses for one activity epoch.
    virtual void advanceUntilGrowth(SimulationInfo* psi);

    //! Updates the network.
    virtual void updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory);

private:
    //! Perform updating neurons for one time step.
    void advanceNeurons(SimulationInfo* psi);

    //! Perform updating synapses for one time step.
    void advanceSynapses(SimulationInfo* psi);
};

#endif // _MULTITHREADEDSIM_H_
