/**
 *      @file XmlRecorder.h
 *
 *      @brief Header file for XmlRecorder.h
 */
//! An implementation for recording spikes history on xml file

/**
 ** \class XmlRecorder.h XmlRecorder.h "XmlRecorder.h"
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
#include "Model.h"
#include <fstream>

class XmlRecorder : public IRecorder
{
public:
    //! THe constructor and destructor
    XmlRecorder(IModel *model, SimulationInfo* sim_info);
    ~XmlRecorder();

    /**
     * Initialize data
     * @param[in] stateOutputFileName       File name to save histories
     */
    virtual void init(const string& stateOutputFileName);

    /*
     * Init radii and rates history matrices with default values
     * @param[in] startRadius       The starting connectivity radius for all neurons
     */
    virtual void initDefaultValues(BGFLOAT startRadius);

    /*
     * Init radii and rates history matrices with current radii and rates
     */
    virtual void initValues();

    /*
     * Get the current radii and rates vlaues
     */
    virtual void getValues();

    /**
     * Terminate process
     */
    virtual void term();

    /**
     * Compile history information in every epoch
     * @param[in] neurons   The entire list of neurons.
     * @param[in] minRadius The minimum possible radius.
     */
    virtual void compileHistories(AllNeurons &neurons, BGFLOAT minRadius);

    /**
     * Save current simulation state to XML
     * @param  neurons the Neuron list to search from.
     **/
    virtual void saveSimState(const AllNeurons &neurons);

private:
    void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, const SimulationInfo *sim_info);

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

    // Struct that holds information about a simulation
    SimulationInfo *m_sim_info;

    // TODO comment
    Model *m_model;
};

#endif // _XMLRECORDER_H_
