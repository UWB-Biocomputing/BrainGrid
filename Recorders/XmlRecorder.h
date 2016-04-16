/**
 *      @file XmlRecorder.h
 *
 *      @brief Header file for XmlRecorder.h
 */
//! An implementation for recording spikes history on xml file

/**
 ** \class XmlRecorder XmlRecorder.h "XmlRecorder.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The XmlRecorder provides a mechanism for recording neuron's layout, spikes history,
 ** and compile history information on xml file:
 **     -# neuron's locations, and type map,
 **     -# individual neuron's spike rate in epochs,
 **     -# network wide burstiness index data in 1s bins,
 **     -# network wide spike count in 10ms bins.
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

#include "IRecorder.h"
#include "Model.h"
#include <fstream>

class XmlRecorder : public IRecorder
{
public:
    //! THe constructor and destructor
    XmlRecorder(const SimulationInfo* sim_info);
    ~XmlRecorder();

    /**
     * Initialize data
     * @param[in] stateOutputFileName       File name to save histories
     */
    virtual void init(const string& stateOutputFileName);

    /**
     * Init radii and rates history matrices with default values
     */
    virtual void initDefaultValues();

    /**
     * Init radii and rates history matrices with current radii and rates
     */
    virtual void initValues();

    /**
     * Get the current radii and rates vlaues
     */
    virtual void getValues();

    /**
     * Terminate process
     */
    virtual void term();

    /**
     * Compile history information in every epoch
     *
     * @param[in] neurons   The entire list of neurons.
     */
    virtual void compileHistories(IAllNeurons &neurons);

    /**
     * Writes simulation results to an output destination.
     *
     * @param  neurons the Neuron list to search from.
     **/
    virtual void saveSimData(const IAllNeurons &neurons);

protected:
    void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, const SimulationInfo *sim_info);

    // a file stream for xml output
    ofstream stateOut;

    // burstiness Histogram goes through the
    VectorMatrix burstinessHist;

    // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
    VectorMatrix spikesHistory;

    // Struct that holds information about a simulation
    const SimulationInfo *m_sim_info;

    // TODO comment
    Model *m_model;
};

