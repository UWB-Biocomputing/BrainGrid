/*
 *      @file Hdf5GrowthRecorder.cpp
 *
 *      @brief An implementation for recording spikes history on hdf5 file
 */
//! An implementation for recording spikes history on hdf5 file

#include "Hdf5GrowthRecorder.h"
#include "AllIFNeurons.h"      // TODO: remove LIF model specific code
#include "ConnGrowth.h"

// hdf5 dataset name
const H5std_string  nameRatesHist("ratesHistory");
const H5std_string  nameRadiiHist("radiiHistory");

//! THe constructor and destructor
Hdf5GrowthRecorder::Hdf5GrowthRecorder(const SimulationInfo* sim_info) :
    Hdf5Recorder(sim_info)
{
}

Hdf5GrowthRecorder::~Hdf5GrowthRecorder()
{
}

/*
 * Initialize data.
 * Create a new hdf5 file with default properties.
 *
 * @param[in] stateOutputFileName	File name to save histories
 */
void Hdf5GrowthRecorder::init(const string& stateOutputFileName)
{
    try
    {
        // create a new file using the default property lists
        stateOut = new H5File( stateOutputFileName, H5F_ACC_TRUNC );

        initDataSet();
    }
    
    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printError();
        return;
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printError();
        return;
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printError();
        return;
    }

    // catch failure caused by the DataType operations
    catch( DataTypeIException error )
    {
        error.printError();
        return;
    }
}

/*
 *  Create data spaces and data sets of the hdf5 for recording histories.
 */
void Hdf5GrowthRecorder::initDataSet()
{
    Hdf5Recorder::initDataSet();

    // create the data space & dataset for rates history
    hsize_t dims[2];
    dims[0] = static_cast<hsize_t>(m_sim_info->maxSteps + 1);
    dims[1] = static_cast<hsize_t>(m_sim_info->totalNeurons);
    DataSpace dsRatesHist(2, dims);
    dataSetRatesHist = new DataSet(stateOut->createDataSet(nameRatesHist, H5_FLOAT, dsRatesHist));

    // create the data space & dataset for radii history
    dims[0] = static_cast<hsize_t>(m_sim_info->maxSteps + 1);
    dims[1] = static_cast<hsize_t>(m_sim_info->totalNeurons);
    DataSpace dsRadiiHist(2, dims);
    dataSetRadiiHist = new DataSet(stateOut->createDataSet(nameRadiiHist, H5_FLOAT, dsRadiiHist));

    // allocate data memories
    ratesHistory = new BGFLOAT[m_sim_info->totalNeurons];
    radiiHistory = new BGFLOAT[m_sim_info->totalNeurons];
}

/*
 * Init radii and rates history matrices with default values
 */
void Hdf5GrowthRecorder::initDefaultValues()
{
    Connections* pConn = m_model->getConnections();
    BGFLOAT startRadius = dynamic_cast<ConnGrowth*>(pConn)->m_growth.startRadius;

    for (int i = 0; i < m_sim_info->totalNeurons; i++)
    {
        radiiHistory[i] = startRadius;
        ratesHistory[i] = 0;
    }

    // write initial radii and rate 
    // because compileHistories function is not called when simulation starts
    writeRadiiRates();
}

/*
 * Init radii and rates history matrices with current radii and rates
 */
void Hdf5GrowthRecorder::initValues()
{
    Connections* pConn = m_model->getConnections();

    for (int i = 0; i < m_sim_info->totalNeurons; i++)
    {
        radiiHistory[i] = (*dynamic_cast<ConnGrowth*>(pConn)->radii)[i];
        ratesHistory[i] = (*dynamic_cast<ConnGrowth*>(pConn)->rates)[i];
    }

    // write initial radii and rate 
    // because compileHistories function is not called when simulation starts
    writeRadiiRates();
}

/*
 * Get the current radii and rates values
 */
void Hdf5GrowthRecorder::getValues()
{
    Connections* pConn = m_model->getConnections();

    for (int i = 0; i < m_sim_info->totalNeurons; i++)
    {
        (*dynamic_cast<ConnGrowth*>(pConn)->radii)[i] = radiiHistory[i];
        (*dynamic_cast<ConnGrowth*>(pConn)->rates)[i] = ratesHistory[i];
    }
}

/*
 * Terminate process
 */
void Hdf5GrowthRecorder::term()
{
    // deallocate all objects
    delete[] ratesHistory;
    delete[] radiiHistory;

    Hdf5Recorder::term();
}

/*
 * Compile history information in every epoch.
 *
 * @param[in] neurons   The entire list of neurons.
 */
void Hdf5GrowthRecorder::compileHistories(IAllNeurons &neurons)
{
    Hdf5Recorder::compileHistories(neurons);

    Connections* pConn = m_model->getConnections();

    BGFLOAT minRadius = dynamic_cast<ConnGrowth*>(pConn)->m_growth.minRadius;
    VectorMatrix& rates = (*dynamic_cast<ConnGrowth*>(pConn)->rates);
    VectorMatrix& radii = (*dynamic_cast<ConnGrowth*>(pConn)->radii);

    // output spikes
    for (int iNeuron = 0; iNeuron < m_sim_info->totalNeurons; iNeuron++)
    {
        // record firing rate to history matrix
        ratesHistory[iNeuron] = rates[iNeuron];

        // Cap minimum radius size and record radii to history matrix
        // TODO: find out why we cap this here.
        if (radii[iNeuron] < minRadius)
            radii[iNeuron] = minRadius;

        // record radius to history matrix
        radiiHistory[iNeuron] = radii[iNeuron];

        DEBUG_MID(cout << "radii[" << iNeuron << ":" << radii[iNeuron] << "]" << endl;)
    }

    writeRadiiRates();
}

/*
 * Incrementaly write radii and rates histories
 */
void Hdf5GrowthRecorder::writeRadiiRates()
{
    try
    {
        // Write radii and rates histories information:
        hsize_t offset[2], count[2];
        hsize_t dimsm[2];
        DataSpace* dataspace;
        DataSpace* memspace;

        // write radii history
        offset[0] = m_sim_info->currentStep;
        offset[1] = 0;
        count[0] = 1;
        count[1] = m_sim_info->totalNeurons;
        dimsm[0] = 1;
        dimsm[1] = m_sim_info->totalNeurons;
        memspace = new DataSpace(2, dimsm, NULL);
        dataspace = new DataSpace(dataSetRadiiHist->getSpace());
        dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
        dataSetRadiiHist->write(radiiHistory, H5_FLOAT, *memspace, *dataspace); 
        delete dataspace;
        delete memspace;

        // write rates history
        offset[0] = m_sim_info->currentStep;
        offset[1] = 0;
        count[0] = 1;
        count[1] = m_sim_info->totalNeurons;
        dimsm[0] = 1;
        dimsm[1] = m_sim_info->totalNeurons;
        memspace = new DataSpace(2, dimsm, NULL);
        dataspace = new DataSpace(dataSetRadiiHist->getSpace());
        dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
        dataSetRatesHist->write(ratesHistory, H5_FLOAT, *memspace, *dataspace); 
        delete dataspace;
        delete memspace;
    }
    
    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printError();
        return;
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printError();
        return;
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printError();
        return;
    }

    // catch failure caused by the DataType operations
    catch( DataTypeIException error )
    {
        error.printError();
        return;
    }
}

