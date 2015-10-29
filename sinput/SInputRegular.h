/**
 ** \brief A class that performs stimulus input (implementation Regular).
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
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

/**
 ** \file SInputRegular.h
 **
 ** \brief Header file for SInputRegular.
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
    virtual void init(IModel* model, AllNeurons &neurons, SimulationInfo* psi);

    //! Terminate process.
    virtual void term(IModel* model, SimulationInfo* psi);

protected:
    //! True if stimuls input is on.
    bool fSInput;

    //! Duration of a pulse in second.
    BGFLOAT duration;

    //! Interval between pulses in second.
    BGFLOAT interval;

    //! The number of time steps for one cycle of a stimulation
    int nStepsCycle;

    //! The time step within a cycle of stimulation
    int nStepsInCycle;

    //! The number of time steps for duration of a pulse.
    int nStepsDuration;

    //! The number of time steps for interval between pulses. 
    int nStepsInterval;

    //! Initial input values
    vector<BGFLOAT> initValues;

    //! Input values, where each entry corresponds with a summationPoint.
    BGFLOAT *values;

    //! Shift values, which determin the synch of stimuli (all 0 when synchronous)
    int *nShiftValues;
};

#endif // _SINPUTREGULAR_H_
