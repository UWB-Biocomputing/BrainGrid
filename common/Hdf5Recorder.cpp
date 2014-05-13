/**
 *      @file Hdf5Recorder.cpp
 *
 *      @brief An implementation for recording spikes history on hdf5 file
 */
//! An implementation for recording spikes history on hdf5 file

#include "Hdf5Recorder.h"
#include "Util.h"

// hdf5 dataset name
const H5std_string  nameBurstHist("burstinessHist");
const H5std_string  nameSpikesHist("spikesHistory");
const H5std_string  nameRatesHist("ratesHistory");
const H5std_string  nameRadiiHist("radiiHistory");
    
const H5std_string  nameXloc("xloc");
const H5std_string  nameYloc("yloc");
const H5std_string  nameNeuronTypes("neuronTypes");
const H5std_string  nameNeuronThresh("neuronThresh");
const H5std_string  nameStarterNeurons("starterNeurons");
const H5std_string  nameTsim("Tsim");
const H5std_string  nameSimulationEndTime("simulationEndTime");

const H5std_string  nameSpikesProbedNeurons("spikesProbedNeurons");
const H5std_string  nameAttrPNUnit("attrPNUint");
const H5std_string  nameProbedNeurons("probedNeurons");

// Load positions of probed neurons list
void LoadPNLayout(string provedNListFileName, vector<int>* pProbedNeuronsLayout);

//! THe constructor and destructor
Hdf5Recorder::Hdf5Recorder(SimulationInfo* sim_info) 
{
}

Hdf5Recorder::~Hdf5Recorder()
{
}

/**
 * Initialize data
 * @param[in] stateOutputFileName	File name to save histories
 * @param[in] probedNListFileName	File name to get locations of probed neurons list
 */
void Hdf5Recorder::init(SimulationInfo* sim_info, const string& stateOutputFileName, const string& probedNListFileName)
{
    try
    {
        // create a new file using the default property lists
        stateOut = new H5File( stateOutputFileName, H5F_ACC_TRUNC );

        // create the data space & dataset for burstiness history
        hsize_t dims[2];
        dims[0] = static_cast<hsize_t>(sim_info->epochDuration * sim_info->maxSteps);
        DataSpace dsBurstHist(1, dims);
        dataSetBurstHist = new DataSet(stateOut->createDataSet(nameBurstHist, PredType::NATIVE_INT, dsBurstHist));

        // create the data space & dataset for spikes history
        dims[0] = static_cast<hsize_t>(sim_info->epochDuration * sim_info->maxSteps * 100);
        DataSpace dsSpikesHist(1, dims);
        dataSetSpikesHist = new DataSet(stateOut->createDataSet(nameSpikesHist, PredType::NATIVE_INT, dsSpikesHist));

        // create the data space & dataset for rates history
        dims[0] = static_cast<hsize_t>(sim_info->maxSteps + 1);
        dims[1] = static_cast<hsize_t>(sim_info->totalNeurons);
        DataSpace dsRatesHist(2, dims);
        dataSetRatesHist = new DataSet(stateOut->createDataSet(nameRatesHist, H5_FLOAT, dsRatesHist));

        // create the data space & dataset for radii histoey
        dims[0] = static_cast<hsize_t>(sim_info->maxSteps + 1);
        dims[1] = static_cast<hsize_t>(sim_info->totalNeurons);
        DataSpace dsRadiiHist(2, dims);
        dataSetRadiiHist = new DataSet(stateOut->createDataSet(nameRadiiHist, H5_FLOAT, dsRadiiHist));

        // create the data space & dataset for xloc & ylo c
        dims[0] = static_cast<hsize_t>(sim_info->totalNeurons);
        DataSpace dsXYloc(1, dims);
        dataSetXloc = new DataSet(stateOut->createDataSet(nameXloc, PredType::NATIVE_INT, dsXYloc));
        dataSetYloc = new DataSet(stateOut->createDataSet(nameYloc, PredType::NATIVE_INT, dsXYloc));

        // create the data space & dataset for neuron types
        dims[0] = static_cast<hsize_t>(sim_info->totalNeurons);
        DataSpace dsNeuronTypes(1, dims);
        dataSetNeuronTypes = new DataSet(stateOut->createDataSet(nameNeuronTypes, PredType::NATIVE_INT, dsNeuronTypes));

        // create the data space & dataset for neuron threashold
        dims[0] = static_cast<hsize_t>(sim_info->totalNeurons);
        DataSpace dsNeuronThresh(1, dims);
        dataSetNeuronThresh = new DataSet(stateOut->createDataSet(nameNeuronThresh, H5_FLOAT, dsNeuronThresh));

        // create the data space & dataset for simulation step duration
        dims[0] = static_cast<hsize_t>(1);
        DataSpace dsTsim(1, dims);
        dataSetTsim = new DataSet(stateOut->createDataSet(nameTsim, H5_FLOAT, dsTsim));

        // create the data space & dataset for simulation end time
        dims[0] = static_cast<hsize_t>(1);
        DataSpace dsSimulationEndTime(1, dims);
        dataSetSimulationEndTime = new DataSet(stateOut->createDataSet(nameSimulationEndTime, H5_FLOAT, dsSimulationEndTime));

        // allocate data memories
        burstinessHist = new int[static_cast<int>(sim_info->epochDuration)];
        spikesHistory = new int[static_cast<int>(sim_info->epochDuration * 100)]; 
        memset(burstinessHist, 0, static_cast<int>(sim_info->epochDuration * sizeof(int)));
        memset(spikesHistory, 0, static_cast<int>(sim_info->epochDuration * 100 * sizeof(int)));
        ratesHistory = new BGFLOAT[sim_info->totalNeurons];
        radiiHistory = new BGFLOAT[sim_info->totalNeurons];

        // initialize probedNeuronsLayout vector
        LoadPNLayout(probedNListFileName, &probedNeuronsLayout);

        // create the data space & dataset for spikes history of probed neurons
        if (probedNeuronsLayout.size() > 0)
        {
            // allocate data for spikesProbedNeurons
            spikesProbedNeurons = new vector<uint64_t>[probedNeuronsLayout.size()];
        }
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
 * Init radii and rates history matrices with default values
 * @param[in] sim_info       Pointer to the simulation information.
 */
void Hdf5Recorder::initValues(SimulationInfo* sim_info)
{
    for (int i = 0; i < sim_info->totalNeurons; i++)
    {
        radiiHistory[i] = sim_info->startRadius;
        ratesHistory[i] = 0;
    }

    // write initial radii and rate 
    // because compileHistories function is not called when simulation starts
    writeRadiiRates(sim_info);
}

/*
 * Init radii and rates history matrices with current radii and rates
 * @param[in] sim_info       Pointer to the simulation information.
 * @param[in] radii	Matrix to hold current radii
 * @param[in] rates	Matrix to hold current rates
 */
void Hdf5Recorder::initValues(SimulationInfo* sim_info, const VectorMatrix& radii, const VectorMatrix& rates)
{
    for (int i = 0; i < sim_info->totalNeurons; i++)
    {
        radiiHistory[i] = radii[i];
        ratesHistory[i] = rates[i];
    }

    // write initial radii and rate 
    // because compileHistories function is not called when simulation starts
    writeRadiiRates(sim_info);
}

/*
 * Get the current radii and rates vlaues
 * @param[in] sim_info       Pointer to the simulation information.
 * @param[out] radii	Current radii values.
 * @param[out] rates	Current rates values.
 */
void Hdf5Recorder::getValues(SimulationInfo* sim_info, VectorMatrix& radii, VectorMatrix& rates)
{
    for (int i = 0; i < sim_info->totalNeurons; i++)
    {
        radii[i] = radiiHistory[i];
        rates[i] = ratesHistory[i];
    }
}

/**
 * Terminate process
 */
void Hdf5Recorder::term(SimulationInfo* sim_info)
{
    // deallocate all objects
    delete[] burstinessHist;
    delete[] spikesHistory;
    delete[] ratesHistory;
    delete[] radiiHistory;

    delete dataSetBurstHist;
    delete dataSetSpikesHist;
    delete dataSetRatesHist;
    delete dataSetRadiiHist;

    if (probedNeuronsLayout.size() > 0)
    {
        delete[] spikesProbedNeurons;
        probedNeuronsLayout.clear();
    }

    delete stateOut;
}

/**
 * Compile history information in every epoch
 * @param[in] sim_info       Pointer to the simulation information.
 * @param[in] rates	Reference to the rates matrix.
 * @param[in] radii	Reference to the radii matrix.
 * @param[in] neurons   The entire list of neurons.
 */
void Hdf5Recorder::compileHistories(const SimulationInfo* sim_info, VectorMatrix& rates, VectorMatrix& radii, AllNeurons &neurons)
{
    unsigned int iProbe = 0;    // index of the probedNeuronsLayout vector
    bool fProbe = false;

    // output spikes
    for (int iNeuron = 0; iNeuron < sim_info->totalNeurons; iNeuron++)
    {
        // true if this is a probed neuron
        fProbe = ((iProbe < probedNeuronsLayout.size()) && (iNeuron == probedNeuronsLayout[iProbe]));

        uint64_t* pSpikes = neurons.spike_history[iNeuron];

        int& spike_count = neurons.spikeCount[iNeuron];
        for (int i = 0; i < spike_count; i++)
        {
            // compile network wide burstiness index data in 1s bins
            int idx1 = static_cast<int>( pSpikes[i] * sim_info->deltaT );
            idx1 -= (sim_info->currentStep - 1) * sim_info->epochDuration;
            assert(idx1 >= 0 && idx1 < sim_info->epochDuration);
            burstinessHist[idx1]++;

            // compile network wide spike count in 10ms bins
            int idx2 = static_cast<int>( pSpikes[i] * sim_info->deltaT * 100 );
            idx2 -= (sim_info->currentStep - 1) * sim_info->epochDuration * 100;
            assert(idx2 >= 0 && idx2 < sim_info->epochDuration * 100);
            spikesHistory[idx2]++;
        }

#if 0
        if (fProbe)
        {
            // compile spikes time of the probed neuron (append spikes time)
            spikesProbedNeurons[iProbe].insert(spikesProbedNeurons[iProbe].end(),(*pSpikes).begin(),(*pSpikes).end());
            iProbe++;
        }
#endif

        // record firing rate to history matrix
        ratesHistory[iNeuron] = rates[iNeuron];

        // Cap minimum radius size and record radii to history matrix
        // TODO: find out why we cap this here.
        if (radii[iNeuron] < sim_info->minRadius)
            radii[iNeuron] = sim_info->minRadius;

        // record radius to history matrix
        radiiHistory[iNeuron] = radii[iNeuron];

        DEBUG_MID(cout << "radii[" << iNeuron << ":" << radii[iNeuron] << "]" << endl;)
    }

    writeRadiiRates(sim_info);

    try
    {
        // write burstiness index
        hsize_t offset[2], count[2];
        hsize_t dimsm[2];
        DataSpace* dataspace;
        DataSpace* memspace;

        offset[0] = (sim_info->currentStep - 1) * sim_info->epochDuration;
        count[0] = sim_info->epochDuration;
        dimsm[0] = sim_info->epochDuration;
        memspace = new DataSpace(1, dimsm, NULL);
        dataspace = new DataSpace(dataSetBurstHist->getSpace());
        dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
        dataSetBurstHist->write(burstinessHist, PredType::NATIVE_INT, *memspace, *dataspace); 
        memset(burstinessHist, 0, static_cast<int>(sim_info->epochDuration * sizeof(int)));
        delete dataspace;
        delete memspace;

        // write network wide spike count in 10ms bins
        offset[0] = (sim_info->currentStep - 1) * sim_info->epochDuration * 100;
        count[0] = sim_info->epochDuration * 100;
        dimsm[0] = sim_info->epochDuration * 100;
        memspace = new DataSpace(1, dimsm, NULL);
        dataspace = new DataSpace(dataSetSpikesHist->getSpace());
        dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
        dataSetSpikesHist->write(spikesHistory, PredType::NATIVE_INT, *memspace, *dataspace); 
        memset(spikesHistory, 0, static_cast<int>(sim_info->epochDuration * 100 * sizeof(int)));
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

/**
 * Incrementaly write radii and rates histories
 * @param[in] sim_info       Pointer to the simulation information.
 */
void Hdf5Recorder::writeRadiiRates(const SimulationInfo* sim_info)
{
    try
    {
        // Write radii and rates histories information:
        hsize_t offset[2], count[2];
        hsize_t dimsm[2];
        DataSpace* dataspace;
        DataSpace* memspace;

        // write radii history
        offset[0] = sim_info->currentStep;
        offset[1] = 0;
        count[0] = 1;
        count[1] = sim_info->totalNeurons;
        dimsm[0] = 1;
        dimsm[1] = sim_info->totalNeurons;
        memspace = new DataSpace(2, dimsm, NULL);
        dataspace = new DataSpace(dataSetRadiiHist->getSpace());
        dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
        dataSetRadiiHist->write(radiiHistory, H5_FLOAT, *memspace, *dataspace); 
        delete dataspace;
        delete memspace;

        // write rates history
        offset[0] = sim_info->currentStep;
        offset[1] = 0;
        count[0] = 1;
        count[1] = sim_info->totalNeurons;
        dimsm[0] = 1;
        dimsm[1] = sim_info->totalNeurons;
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

/**
 * Save current simulation state to XML
 * @param[in] sim_info       Pointer to the simulation information.
 * @param[in] neuronTypes	Neuron types: INH or EXC
 * @param[in] starterNeurons	Starter neurons matrix
 * @param[in] neuronThresh	Neuron thresold
 **/
void Hdf5Recorder::saveSimState(const SimulationInfo* sim_info, VectorMatrix& neuronTypes, VectorMatrix& starterNeurons, VectorMatrix& neuronThresh)
{
    try
    {
        // neuron locations matrices
        int* xloc = new int[sim_info->totalNeurons];
        int* yloc = new int[sim_info->totalNeurons];

        // Initialize neurons
        for (int i = 0; i < sim_info->totalNeurons; i++)
        {
            xloc[i] = i % sim_info->width;
            yloc[i] = i / sim_info->width;
        }

        // Write the location matrices
        dataSetXloc->write(xloc, PredType::NATIVE_INT);
        dataSetYloc->write(yloc, PredType::NATIVE_INT);

        delete[] xloc;
        delete[] yloc;

        int* iNeuronTypes = new int[sim_info->totalNeurons];
        for (int i = 0; i < sim_info->totalNeurons; i++)
        {
            iNeuronTypes[i] = neuronTypes[i];
        }
        dataSetNeuronTypes->write(iNeuronTypes, PredType::NATIVE_INT);
        delete[] iNeuronTypes;

        if (starterNeurons.Size() > 0)
        {
            // create the data space & dataset for starter neurons
            hsize_t dims[2];
            dims[0] = static_cast<hsize_t>(starterNeurons.Size());
            DataSpace dsStarterNeurons(1, dims);
            dataSetStarterNeurons = new DataSet(stateOut->createDataSet(nameStarterNeurons, PredType::NATIVE_INT, dsStarterNeurons));

            int* iStarterNeurons = new int[starterNeurons.Size()];
            for (int i = 0; i < starterNeurons.Size(); i++)
            {
                iStarterNeurons[i] = starterNeurons[i];
            }
            dataSetStarterNeurons->write(iStarterNeurons, PredType::NATIVE_INT);
            delete[] iStarterNeurons;
            delete dataSetStarterNeurons;
        }

        if (probedNeuronsLayout.size() > 0)
        {
            // create the data space & dataset for probed neurons
            hsize_t dims[2];
            dims[0] = static_cast<hsize_t>(probedNeuronsLayout.size());
            DataSpace dsProbedNeurons(1, dims);
            dataSetProbedNeurons = new DataSet(stateOut->createDataSet(nameProbedNeurons, PredType::NATIVE_INT, dsProbedNeurons));

            int* iProbedNeurons = new int[probedNeuronsLayout.size()];
            for (unsigned int i = 0; i < probedNeuronsLayout.size(); i++)
            {
                iProbedNeurons[i] = probedNeuronsLayout[i];
            }
            dataSetProbedNeurons->write(iProbedNeurons, PredType::NATIVE_INT);
            delete[] iProbedNeurons;
            delete dataSetProbedNeurons;

            // create the data space & dataset for spikes of probed neurons
            unsigned int max_size = 0;
            for (unsigned int i = 0; i < probedNeuronsLayout.size(); i++)
            {
                max_size = (max_size > spikesProbedNeurons[i].size()) ? max_size : spikesProbedNeurons[i].size();
            }
            dims[0] = static_cast<hsize_t>(max_size);
            dims[1] = static_cast<hsize_t>(probedNeuronsLayout.size());
            DataSpace dsSpikesProbedNeurons(2, dims);
            dataSetSpikesProbedNeurons = new DataSet(stateOut->createDataSet(nameSpikesProbedNeurons, PredType::NATIVE_UINT64, dsSpikesProbedNeurons));

            // write it!
            for (unsigned int i = 0; i < probedNeuronsLayout.size(); i++)
            {
                hsize_t offset[2], count[2];
                hsize_t dimsm[2];
                DataSpace* dataspace;
                DataSpace* memspace;

                offset[0] = 0;
                offset[1] = i;
                count[0] = spikesProbedNeurons[i].size();
                count[1] = 1;
                dimsm[0] = spikesProbedNeurons[i].size();
                dimsm[1] = 1;
                memspace = new DataSpace(2, dimsm, NULL);
                dataspace = new DataSpace(dataSetSpikesProbedNeurons->getSpace());
                dataspace->selectHyperslab(H5S_SELECT_SET, count, offset);
                dataSetSpikesProbedNeurons->write(static_cast<uint64_t*>(&(spikesProbedNeurons[i][0])), PredType::NATIVE_UINT64, *memspace, *dataspace); 
                delete dataspace;
                delete memspace;
            }

            // Create the data space for the attribute (unit of the spikes of probed neurons in second).
            dims[0] = 1;
            DataSpace dsAttrPNUnit(1, dims);

            // Create a dataset attribute. 
            Attribute attribute = dataSetSpikesProbedNeurons->createAttribute(nameAttrPNUnit, H5_FLOAT, dsAttrPNUnit, PropList::DEFAULT);
     
            // Write the attribute data. 
            attribute.write(H5_FLOAT, &(sim_info->deltaT));

            delete dataSetSpikesProbedNeurons;
        }

        // Write neuron thresold
        // Write neuron thresold
        BGFLOAT* fNeuronThresh = new BGFLOAT[sim_info->totalNeurons];
        for (int i = 0; i < sim_info->totalNeurons; i++)
        {
            fNeuronThresh[i] = neuronThresh[i];
        }
        dataSetNeuronThresh->write(fNeuronThresh, H5_FLOAT);
        delete[] fNeuronThresh;
    
        // write time between growth cycles
        dataSetTsim->write(&sim_info->epochDuration, H5_FLOAT);
        delete dataSetTsim;

        // write simulation end time
        BGFLOAT endtime = g_simulationStep * sim_info->deltaT;
        dataSetSimulationEndTime->write(&endtime, H5_FLOAT);
        delete dataSetSimulationEndTime;
    }

    // catch failure caused by the DataSet operations
    catch (DataSetIException error)
    {
        error.printError();
        return;
    }

    // catch failure caused by the DataSpace operations
    catch (DataSpaceIException error)
    {
        error.printError();
        return;
    }
}

//extern void getValueList(const char *valString, vector<int>* pList);

// Load positions of probed neurons list
void LoadPNLayout(string provedNListFileName, vector<int>* pProbedNeuronsLayout)
{
    TiXmlDocument simDoc( provedNListFileName.c_str( ) );
    if (!simDoc.LoadFile( ))
    {
        cerr << "Failed loading positions of probed neurons list file " << provedNListFileName << ":" << "\n\t"
                            << simDoc.ErrorDesc( ) << endl;
        cerr << " error: " << simDoc.ErrorRow( ) << ", " << simDoc.ErrorCol( ) << endl;
        return;
    }
    TiXmlNode* temp;
    if (( temp = simDoc.FirstChildElement( "P" ) ) == NULL)
    {
        cerr << "Could not find <P> in positions of probed neurons list file " << provedNListFileName << endl;
    }
    getValueList(temp->ToElement()->GetText(), pProbedNeuronsLayout);

    cout << "Layout parameters:" << endl;

    cout << "\tProbed neuron positions: ";
    for (size_t i = 0; i < pProbedNeuronsLayout->size(); i++)
    {
        cout << (*pProbedNeuronsLayout)[i] << " ";
    }
    cout << endl;
}
