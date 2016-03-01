/**
 *      @file IRecorder.h
 *
 *      @brief Header file for IRecorder.h
 */
//! An interface for recording spikes history

/**
 ** \class IRecorder IRecorder.h "IRecorder.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The IRecorder provides an interface for recording spikes history.
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

#ifndef _IRECORDER_H_
#define _IRECORDER_H_

#include "Global.h"
#include "SimulationInfo.h"
#include "AllSpikingNeurons.h"

class IRecorder
{
public:
    virtual ~IRecorder() {}

    /**
     * Initialize data
     *
     * @param[in] stateOutputFileName       File name to save histories
     */
    virtual void init(const string& stateOutputFileName) = 0;

    /**
     * Init radii and rates history matrices with default values
     */
    virtual void initDefaultValues() = 0;

    /**
     * Init radii and rates history matrices with current radii and rates
     */
    virtual void initValues() = 0;

    /**
     * Get the current radii and rates vlaues
     */
    virtual void getValues() = 0;

    /**
     * Terminate process
     */
    virtual void term() = 0;

    /**
     * Compile history information in every epoch
     *
     * @param[in] neurons   The entire list of neurons.
     */
    virtual void compileHistories(IAllNeurons &neurons) = 0;

    /**
     * Writes simulation results to an output destination.
     *
     * @param[in] neurons   The entire list of neurons.
     **/
    virtual void saveSimData(const IAllNeurons &neurons) = 0;
};

#endif // _IRECORDER_H_
