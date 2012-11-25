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

class ISimulation
{
public:
    virtual ~ISimulation() {}

    /**
     * Initialize data
     * @param psi
     * @param xloc
     * @param yloc
     */
    virtual void init(SimulationInfo* psi, VectorMatrix& xloc, VectorMatrix& yloc) = 0;

    /**
     * Terminate process
     * @param psi
     */
    virtual void term(SimulationInfo* psi) = 0;

    /** 
     * Set initial radii data
     * @param[in] newRadii  Radii data to set
     */ 
    virtual void initRadii(VectorMatrix& newRadii) = 0;

    /**
     * Performs updating neurons and synapses for one activity epoch.
     * @param psi
     */
    virtual void advanceUntilGrowth(SimulationInfo* psi) = 0;

    /**
     * Updates synapses' weight between neurons.
     * @param psi
     * @param radiiHistory
     * @param ratesHistory
     */
    virtual void updateNetwork(SimulationInfo* psi, CompleteMatrix& radiiHistory, CompleteMatrix& ratesHistory) = 0;
};

#endif // _ISIMULATION_H_
