/**
 *      @file XmlRecorder.h
 *
 *      @brief Header file for XmlRecorder.h
 */
//! An implementation for recording spikes history on xml file

/**
 ** \class XmlRecorder.h XmlRecorder.h.h "XmlRecorder.h.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The XmlRecorder provides a mechanism for recording spikes history,
 ** and compile history information on xml file:
 ** 	(1) individual neuron's spike rate in epochs,
 **	(2) burstiness index data in 1s bins,
 **     (3) network wide spike count in 10ms bins.
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

#ifndef _XMLRECORDER_H_
#define _XMLRECORDER_H_

#include "IRecorder.h"
#include <fstream>

class XmlRecorder : public IRecorder
{
public:
    //! THe constructor and destructor
    XmlRecorder(SimulationInfo* psi);
    ~XmlRecorder();

    /**
     * Initialize data
     * @param[in] stateOutputFileName       File name to save histories
     * @param[in] probedNListFileName       File name to get locations of probed neurons list
     */
    virtual void init(SimulationInfo* psi, const string& stateOutputFileName, const string& probedNListFileName);

    /*
     * Init radii and rates history matrices with default values
     * @param[in] psi       Pointer to the simulation information.
     */
    virtual void initValues(SimulationInfo* psi);

    /*
     * Init radii and rates history matrices with current radii and rates
     * @param[in] psi       Pointer to the simulation information.
     * @param[in] radii     Matrix to hold current radii.
     * @param[in] rates     Matrix to hold current rates.
     */
    virtual void initValues(SimulationInfo* psi, const VectorMatrix& radii, const VectorMatrix& rates);

    /*
     * Get the current radii and rates vlaues
     * @param[in] psi       Pointer to the simulation information.
     * @param[out] radii    Current radii values
     * @param[out] rates    Current rates values
     */
    virtual void getValues(SimulationInfo* psi, VectorMatrix& radii, VectorMatrix& rates);

    /**
     * Terminate process
     */
    virtual void term(SimulationInfo* psi);

    /**
     * Compile history information in every epoch
     * @param[in] psi       Pointer to the simulation information.
     * @param[in] rates     Reference to the rates matrix.
     * @param[in] radii     Reference to the radii matrix.
     * @param[in] neurons   The entire list of neurons.
     */
    virtual void compileHistories(const SimulationInfo* psi, VectorMatrix& rates, VectorMatrix& radii, AllNeurons &neurons);

    /**
     * Save current simulation state to XML
     * @param[in] psi       Pointer to the simulation information.
     * @param[in] neuronTypes       Neuron types: INH or EXC
     * @param[in] starterNeurons    Starter neurons matrix
     * @param[in] neuronThresh      Neuron thresold
     **/
    virtual void saveSimState(const SimulationInfo* psi, VectorMatrix& neuronTypes, VectorMatrix& starterNeurons, VectorMatrix& neuronThresh);

private:
    // a file stream for xml output
    ofstream stateOut;

    // burstiness Histogram goes through the
    VectorMatrix burstinessHist;

    // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
    VectorMatrix spikesHistory;

    // track radii
    CompleteMatrix radiiHistory;

    // track firing rate
    CompleteMatrix ratesHistory;
};

#endif // _XMLRECORDER_H_
