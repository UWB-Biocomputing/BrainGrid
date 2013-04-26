/**
 *      @file ISimulation.h
 *
 *      @brief Header file for ISimulation.
 */
//! An interface for simulation classes.

/**
 ** \class ISimulation ISimulation.h "ISimulation.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The ISimulation provides an interface for simulation classes.
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

#ifndef _ISIMULATION_H_
#define _ISIMULATION_H_

#include "SimulationInfo.h"
#include "matrix/VectorMatrix.h"

//! Pure virtual interface that can be modified to make a different simulator
class ISimulation
{
public:
    virtual ~ISimulation() {}

    //! Initialize data.
    virtual void init(SimulationInfo* psi, VectorMatrix& xloc, VectorMatrix& yloc) = 0;

    //! Terminate process.
    virtual void term(SimulationInfo* psi) = 0;

// NETWORK MODEL VARIABLES NMV-BEGIN {
    //! Initialize radii
    virtual void initRadii(VectorMatrix& newRadii) = 0;
// } NMV-END

    //! Perform updating neurons for one time step.
    virtual void advanceUntilGrowth(SimulationInfo* psi) = 0;

    //! Perform updating synapses for one time step.
    virtual void updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory) = 0;

	//! Returns a type of Neuron to be used in the Network
	virtual INeuron* returnNeuron() = 0;
};

#endif // _ISIMULATION_H_
