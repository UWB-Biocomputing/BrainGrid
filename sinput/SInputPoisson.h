/**
 ** \brief A class that performs stimulus input (implementation Poisson).
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
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

/**
 ** \file SInputPoisson.h
 **
 ** \brief Header file for SInputPoisson.
 **/

#pragma once

#ifndef _SINPUTPOISSON_H_
#define _SINPUTPOISSON_H_

#include "ISInput.h"
#include "AllSynapses.h"

class SInputPoisson : public ISInput
{
public:
    //! The constructor for SInputPoisson.
    SInputPoisson();
    ~SInputPoisson();

    //! Initialize data.
    virtual void init(Model* model, SimulationInfo* psi, TiXmlElement* parms);

    //! Terminate process.
    virtual void term(Model* model, SimulationInfo* psi);

protected:
    //! True if stimuls input is on.
    bool fSInput;

    //! synapse weight
    BGFLOAT weight;

    //! inverse firing rate
    BGFLOAT lambda;

    //! interval counter
    int* nISIs;
};

#endif // _SINPUTPOISSON_H_
