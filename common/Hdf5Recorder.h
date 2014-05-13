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
    Hdf5Recorder(SimulationInfo* psi);
    ~Hdf5Recorder();

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
    /**
     * Incrementaly write radii and rates histories
     * @param[in] psi       Pointer to the simulation information.
     */
    void writeRadiiRates(const SimulationInfo* psi);

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

    // positions of probed neurons list
    vector<int> probedNeuronsLayout;
};

#endif // _HD5RECORDER_H_
