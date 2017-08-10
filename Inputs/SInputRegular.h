/**
 *      @file SInputRegular.h
 *
 *      @brief A class that performs stimulus input (implementation Regular).
 */

/**
 **
 ** \class SInputRegular SInputRegular.h "SInputRegular.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The SInputRegular performs providing stimulus input to the network for each time step.
 ** Inputs are series of current pulses, which are characterized by a duration, an interval
 ** and input values.
 **
 ** This class is the base class of GpuSInputRegular and HostSInputRegular.
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

#ifndef _SINPUTREGULAR_H_
#define _SINPUTREGULAR_H_

#include "ISInput.h"

class SInputRegular : public ISInput
{
public:
    //! The constructor for SInputRegular.
    SInputRegular(SimulationInfo* psi, TiXmlElement* parms);
    ~SInputRegular();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo);

    //! Terminate process.
    virtual void term(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo);

protected:
    //! True if stimuls input is on.
    bool m_fSInput;

    //! Duration of a pulse in second.
    BGFLOAT m_duration;

    //! Interval between pulses in second.
    BGFLOAT m_interval;

    //! The number of time steps for one cycle of a stimulation
    int m_nStepsCycle;

    //! The time step within a cycle of stimulation
    int m_nStepsInCycle;

    //! The number of time steps for duration of a pulse.
    int m_nStepsDuration;

    //! The number of time steps for interval between pulses. 
    int m_nStepsInterval;

    //! Initial input values
    vector<BGFLOAT> m_initValues;

    //! Input values, where each entry corresponds with a summationPoint.
    BGFLOAT *m_values;

    //! Shift values, which determin the synch of stimuli (all 0 when synchronous)
    int *m_nShiftValues;
};

#endif // _SINPUTREGULAR_H_
