/**
 *      @file GpuSInputRegular.h
 *
 *      @brief A class that performs stimulus input (implementation Regular on GPU).
 */

/**
 **
 ** \class GpuSInputRegular GpuSInputRegular.h "GpuSInputRegular.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The GpuSInputRegular performs providing stimulus input to the network for each time step on GPU.
 ** Inputs are series of current pulses, which are characterized by a duration, an interval
 ** and input values.
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

#ifndef _GPUSINPUTREGULAR_H_
#define _GPUSINPUTREGULAR_H_

#include "SInputRegular.h"

class GpuSInputRegular : public SInputRegular
{
public:
    //! The constructor for SInputRegular.
    GpuSInputRegular(SimulationInfo* psi, TiXmlElement* parms);
    ~GpuSInputRegular();

    //! Initialize data.
    virtual void init(SimulationInfo* psi);

    //! Terminate process.
    virtual void term(SimulationInfo* psi);

    //! Process input stimulus for each time step.
    virtual void inputStimulus(SimulationInfo* psi);
};

//! Device function that processes input stimulus for each time step.
#if defined(__CUDACC__)
extern __global__ void inputStimulusDevice( int n, BGFLOAT* summationPoint_d, BGFLOAT* initValues_d, int* nShiftValues_d, int nStepsInCycle, int nStepsCycle, int nStepsDuration );
#endif

#endif // _GPUSINPUTREGULAR_H_
