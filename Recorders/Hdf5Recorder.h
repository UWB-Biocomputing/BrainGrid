/**
 *      @file Hdf5Recorder.h
 *
 *      @brief Header file for Hdf5Recorder.h
 */
//! An implementation for recording spikes history on hdf5 file

/**
 ** @class Hdf5Recorder Hdf5Recorder.h "Hdf5Recorder.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The Hdf5Recorder provides a mechanism for recording neuron's layout, spikes history,
 ** and compile history information on hdf5 file:
 **     -# neuron's locations, and type map,
 **     -# individual neuron's spike rate in epochs,
 **     -# network wide burstiness index data in 1s bins,
 **     -# network wide spike count in 10ms bins.
 **
 ** Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed 
 ** to store and organize large amounts of data.
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
#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

#ifdef SINGLEPRECISION
#define H5_FLOAT PredType::NATIVE_FLOAT
#else
#define H5_FLOAT PredType::NATIVE_DOUBLE
#endif

class Hdf5Recorder : public IRecorder
{
public:
    //! THe constructor and destructor
    Hdf5Recorder(const SimulationInfo* sim_info);
    ~Hdf5Recorder();

    /**
     * Initialize data
     *
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
    virtual void initDataSet();

    void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, const SimulationInfo *sim_info);

    // hdf5 file identifier
    H5File* stateOut;

    // hdf5 file dataset
    DataSet* dataSetBurstHist;
    DataSet* dataSetSpikesHist;

    DataSet* dataSetXloc;
    DataSet* dataSetYloc;
    DataSet* dataSetNeuronTypes;
    DataSet* dataSetNeuronThresh;
    DataSet* dataSetStarterNeurons;
    DataSet* dataSetTsim;
    DataSet* dataSetSimulationEndTime;

    DataSet* dataSetSpikesProbedNeurons;
    DataSet* dataSetProbedNeurons;

    // burstiness Histogram goes through the
    int* burstinessHist;

    // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
    int* spikesHistory;

    // track spikes count of probed neurons
    vector<uint64_t>* spikesProbedNeurons;

    // Struct that holds information about a simulation
    const SimulationInfo *m_sim_info;

    // TODO comment
    Model *m_model;
};

