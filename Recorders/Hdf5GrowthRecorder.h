/**
 *      @file Hdf5GrowthRecorder.h
 *
 *      @brief An implementation for recording spikes history on hdf5 file
 */

/**
 ** @class Hdf5GrowthRecorder Hdf5GrowthRecorder.h "Hdf5GrowthRecorder.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The Hdf5GrowthRecorder provides a mechanism for recording neuron's layout, spikes history,
 ** and compile history information on hdf5 file:
 **     -# neuron's locations, and type map,
 **     -# individual neuron's spike rate in epochs,
 **     -# network wide burstiness index data in 1s bins,
 **     -# network wide spike count in 10ms bins,
 **     -# individual neuron's radius history of every epoch.
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

#include "Hdf5Recorder.h"
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

class Hdf5GrowthRecorder : public Hdf5Recorder
{
public:
    //! THe constructor and destructor
    Hdf5GrowthRecorder(const SimulationInfo* sim_info);
    ~Hdf5GrowthRecorder();

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
     * @param[in] neurons   The entire list of neurons.
     */
    virtual void compileHistories(IAllNeurons &neurons);

protected:
    virtual void initDataSet();

    /**
     * Incrementaly write radii and rates histories
     */
    void writeRadiiRates();

    // hdf5 file dataset
    DataSet* dataSetRatesHist;
    DataSet* dataSetRadiiHist;

    // track radii
    BGFLOAT* radiiHistory;

    // track firing rate
    BGFLOAT* ratesHistory;
};

