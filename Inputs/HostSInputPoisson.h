/**
 *      @file HostSInputPoisson.h
 *
 *      @brief A class that performs stimulus input (implementation Poisson).
 */

/**
 **
 ** \class HostSInputPoisson HostSInputPoisson.h "HostSInputPoisson.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The HostSInputPoisson performs providing stimulus input to the network for each time step.
 ** In this version, a layer of synapses are added, which accept external spike trains.
 ** Each synapse gets an indivisual spike train (Poisson distribution) characterized
 ** by mean firing rate, and each synapse has individual weight value.
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

#ifndef _HOSTSINPUTPOISSON_H_
#define _HOSTSINPUTPOISSON_H_

#include "SInputPoisson.h"

class HostSInputPoisson : public SInputPoisson
{
public:
    // The constructor for HostSInputPoisson.
    HostSInputPoisson(SimulationInfo* psi, TiXmlElement* parms);
    ~HostSInputPoisson();

    // Initialize data.
    virtual void init(SimulationInfo* psi);

    // Terminate process.
    virtual void term(SimulationInfo* psi);

    // Process input stimulus for each time step.
    virtual void inputStimulus(SimulationInfo* psi);

private:
};

#endif // _HOSTSINPUTPOISSON_H_
