/**
 *      @file SInput.h
 *
 *      @brief A class that performs stimulus input
 */

/**
 **
 ** \class SInput SInput.h "SInput.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The SInput performs providing stimulus input to the network for each time step.
 ** In this version, a layer of synapses are added, which accept external spike trains.
 ** Each synapse gets an indivisual spike train.
 **
 ** This class is the base class of other stimulus input classes.
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

#ifndef _SINPUT_H_
#define _SINPUT_H_

#include "ISInput.h"

class SInput : public ISInput
{
public:
    //! The constructor foer SInput
    SInput(SimulationInfo* psi, BGFLOAT weight, vector<BGFLOAT> const &maskIndex);
    virtual ~SInput();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo);

    //! Terminate process.
    virtual void term(SimulationInfo* psi, vector<ClusterInfo *> const&vtClrInfo);

    // Process input stimulus for each time step.
    virtual void advanceSInputState(const ClusterInfo *pci, int iStep);

protected:
    //! True if stimuls input is on.
    bool m_fSInput;

    //! synapse weight
    BGFLOAT m_weight;

    //! Maximum number of synapses per neuron (will be 1)
    int m_maxSynapsesPerNeuron;

    //! Masks for stimulus input
    bool* m_masks;

    //! interval counter
    int* m_nISIs;
};

#endif // _SINPUT_H_
