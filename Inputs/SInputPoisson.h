/**
 *      @file SInputPoisson.h
 *
 *      @brief A class that performs stimulus input (implementation Poisson).
 */

/**
 **
 ** \class SInputPoisson SInputPoisson.h "SInputPoisson.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The SInputPoisson performs providing stimulus input to the network for each time step.
 ** In this version, a layer of synapses are added, which accept external spike trains. 
 ** Each synapse gets an indivisual spike train (Poisson distribution) characterized 
 ** by mean firing rate, and each synapse has individual weight value. 
 **
 ** This class is the base class of GpuSInputPoisson and HostSInputPoisson.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other 
 ** work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

#pragma once

#ifndef _SINPUTPOISSON_H_
#define _SINPUTPOISSON_H_

#include "SInput.h"
#include "AllDSSynapses.h"

class SInputPoisson : public SInput
{
public:
    //! The constructor for SInputPoisson.
    SInputPoisson(SimulationInfo* psi, BGFLOAT fr_mean, BGFLOAT weight, vector<BGFLOAT> &maskIndex);
    virtual ~SInputPoisson();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo);

    //! Terminate process.
    virtual void term(SimulationInfo* psi, vector<ClusterInfo *> const&vtClrInfo);

protected:
    //! inverse firing rate
    BGFLOAT m_lambda;
};

#endif // _SINPUTPOISSON_H_
