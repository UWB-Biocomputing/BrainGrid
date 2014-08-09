/**
 ** \brief A class that performs stimulus input (implementation Regular).
 **
 ** \class HostSInputRegular HostSInputRegular.h "HostSInputRegular.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The HostSInputRegular.hSInputRegular performs providing stimulus input to the network for each time step.
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
 ** \file HostSInputRegular.h
 **
 ** \brief Header file for HostSInputRegular.
 **/

#pragma once

#ifndef _HOSTSINPUTREGULAR_H_
#define _HOSTSINPUTREGULAR_H_

#include "SInputRegular.h"

class HostSInputRegular : public SInputRegular
{
public:
    //! The constructor for HostSInputRegular.
    HostSInputRegular();
    ~HostSInputRegular();

    //! Initialize data.
    virtual void init(SimulationInfo* psi, TiXmlElement* parms);

    //! Terminate process.
    virtual void term();

    //! Process input stimulus for each time step.
    virtual void inputStimulus(SimulationInfo* psi, BGFLOAT* summationPoint);

private:
};

#endif // _HOSTSINPUTREGULAR_H_
