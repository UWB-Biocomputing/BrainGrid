/**
 *      @file Hdf5Recorder.h
 *
 *      @brief Header file for Hdf5Recorder.h
 */
//! An implementation for recording spikes history on hdf5 file

/**
 ** \class Hdf5Recorder.h Hdf5Recorder.h.h "Hdf5Recorder.h.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The Hdf5Recorder provides a mechanism for recording spikes history,
 ** and compile history information on hdf5 file:
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

#ifndef _HD5RECORDER_H_
#define _HD5RECORDER_H_

#include "IRecorder.h"
#include "LIFModel.h"
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
    Hdf5Recorder(Model *model, SimulationInfo* sim_info);
    ~Hdf5Recorder();

    /**
     * Initialize data
     * @param[in] stateOutputFileName       File name to save histories
     */
    virtual void init(const string& stateOutputFileName);

    /*
     * Init radii and rates history matrices with default values
     */
    virtual void initDefaultValues();

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
     */
    virtual void compileHistories(const AllNeurons &neurons);

    /**
     * Save current simulation state to XML
     * @param  neurons the Neuron list to search from.
     **/
    virtual void saveSimState(const AllNeurons &neurons);

private:
    /**
     * Incrementaly write radii and rates histories
     */
    void writeRadiiRates();

    // hdf5 file identifier
    H5File* stateOut;

    // hdf5 file dataset
    DataSet* dataSetBurstHist;
    DataSet* dataSetSpikesHist;
    DataSet* dataSetRatesHist;
    DataSet* dataSetRadiiHist;

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

    // track radii
    BGFLOAT* radiiHistory;

    // track firing rate
    BGFLOAT* ratesHistory;

    // track spikes count of probed neurons
    vector<uint64_t>* spikesProbedNeurons;

    // Struct that holds information about a simulation
    SimulationInfo *m_sim_info;

    // TODO comment
    LIFModel *m_model;
};

#endif // _HD5RECORDER_H_
