/**
 *      @file IRecorder.h
 *
 *      @brief Header file for IRecorder.h
 */
//! An interface for recording spikes history

/**
 ** \class IRecorder.h IRecorder.h.h "IRecorder.h.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The IRecorder provides an interface for recording spikes history.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** This simulator is a rewrite of CSIM (2006) and other work (Stiber and Kawasaki (2007?))
 **
 **
 **     @author Fumitaka Kawasaki
 **/

#pragma once

#ifndef _IRECORDER_H_
#define _IRECORDER_H_

#include "Global.h"
#include "SimulationInfo.h"
#include "AllNeurons.h"

class IRecorder
{
public:
    virtual ~IRecorder() {}

    /**
     * Initialize data
     * @param[in] stateOutputFileName       File name to save histories
     * @param[in] probedNListFileName       File name to get locations of probed neurons list
     */
    virtual void init(SimulationInfo* sim_info, const string& stateOutputFileName, const string& probedNListFileName) = 0;

    /*
     * Init radii and rates history matrices with default values
     * @param[in] sim_info       Pointer to the simulation information.
     */
    virtual void initValues(SimulationInfo* sim_info) = 0;

    /*
     * Init radii and rates history matrices with current radii and rates
     * @param[in] sim_info       Pointer to the simulation information.
     * @param[in] radii     Matrix to hold current radii
     * @param[in] rates     Matrix to hold current rates
     */
    virtual void initValues(SimulationInfo* sim_info, const VectorMatrix& radii, const VectorMatrix& rates) = 0;

    /*
     * Get the current radii and rates vlaues
     * @param[in] sim_info       Pointer to the simulation information.
     * @param[out] radii    Current radii values
     * @param[out] rates    Current rates values
     */
    virtual void getValues(SimulationInfo* sim_info, VectorMatrix& radii, VectorMatrix& rates) = 0;

    /**
     * Terminate process
     */
    virtual void term(SimulationInfo* sim_info) = 0;

    /**
     * Compile history information in every epoch
     * @param[in] sim_info       Pointer to the simulation information.
     * @param[in] rates     Reference to the rates matrix.
     * @param[in] radii     Reference to the radii matrix.
     * @param[in] neurons   The entire list of neurons.
     */
    virtual void compileHistories(const SimulationInfo* sim_info, VectorMatrix& rates, VectorMatrix& radii, AllNeurons &neurons) = 0;

    /**
     * Save current simulation state to XML
     * @param[in] sim_info       Pointer to the simulation information.
     * @param[in] neuronTypes       Neuron types: INH or EXC
     * @param[in] starterNeurons    Starter neurons matrix
     * @param[in] neuronThresh      Neuron thresold
     **/
    virtual void saveSimState(const SimulationInfo* sim_info, VectorMatrix& neuronTypes, VectorMatrix& starterNeurons, VectorMatrix& neuronThresh) = 0;
};

#endif // _IRECORDER_H_
