/**
 *      @file XmlRecorder.cpp
 *
 *      @brief An implementation for recording spikes history on xml file
 */
//! An implementation for recording spikes history on xml file

#include "XmlRecorder.h"

//! THe constructor and destructor
XmlRecorder::XmlRecorder(SimulationInfo* sim_info) :
        burstinessHist("complete", "const", 1, static_cast<int>(sim_info->epochDuration * sim_info->maxSteps), 0),
        spikesHistory("complete", "const", 1, static_cast<int>(sim_info->epochDuration * sim_info->maxSteps * 100), 0),
        ratesHistory("complete", "const", static_cast<int>(sim_info->maxSteps + 1), sim_info->totalNeurons),
        radiiHistory("complete", "const", static_cast<int>(sim_info->maxSteps + 1), sim_info->totalNeurons)
{
}

XmlRecorder::~XmlRecorder()
{
}

/**
 * Initialize data
 * @param[in] stateOutputFileName	File name to save histories
 * @param[in] probedNListFileName       File name to get locations of probed neurons list
 */
void XmlRecorder::init(SimulationInfo* sim_info, const string& stateOutputFileName, const string& probedNListFileName)
{
    stateOut.open( stateOutputFileName.c_str( ) );

}

/*
 * Init radii and rates history matrices with default values
 * @param[in] sim_info       Pointer to the simulation information.
 */
void XmlRecorder::initValues(SimulationInfo* sim_info)
{
    for (int i = 0; i < sim_info->totalNeurons; i++)
    {
        radiiHistory(0, i) = sim_info->startRadius;
        ratesHistory(0, i) = 0;
    }
}

/*
 * Init radii and rates history matrices with current radii and rates
 * @param[in] sim_info       Pointer to the simulation information.
 * @param[in] radii	Matrix to hold current radii
 * @param[in] rates	Matrix to hold current rates
 */
void XmlRecorder::initValues(SimulationInfo* sim_info, const VectorMatrix& radii, const VectorMatrix& rates)
{
    for (int i = 0; i < sim_info->totalNeurons; i++)
    {
        radiiHistory(0, i) = radii[i];
        ratesHistory(0, i) = rates[i];
    }
}

/*
 * Get the current radii and rates vlaues
 * @param[in] sim_info       Pointer to the simulation information.
 * @param[out] radii	Current radii values.
 * @param[out] rates	Current rates values.
 */
void XmlRecorder::getValues(SimulationInfo* sim_info, VectorMatrix& radii, VectorMatrix& rates)
{
    for (int i = 0; i < sim_info->totalNeurons; i++)
    {
        radii[i] = radiiHistory(sim_info->currentStep, i);
        rates[i] = ratesHistory(sim_info->currentStep, i);
    }
}

/**
 * Terminate process
 */
void XmlRecorder::term(SimulationInfo* sim_info)
{
    stateOut.close();
}

/**
 * Compile history information in every epoch
 * @param[in] sim_info  Pointer to the simulation information.
 * @param[in] rates	Reference to the rates matrix.
 * @param[in] radii	Reference to the radii matrix.
 * @param[in] neurons 	The entire list of neurons.
 */
void XmlRecorder::compileHistories(const SimulationInfo* sim_info, VectorMatrix& rates, VectorMatrix& radii, AllNeurons &neurons)
{
    // output spikes
    for (int iNeuron = 0; iNeuron < sim_info->totalNeurons; iNeuron++)
    {
        uint64_t* pSpikes = neurons.spike_history[iNeuron];

        int& spike_count = neurons.spikeCount[iNeuron];
        for (unsigned int i = 0; i < spike_count; i++)
        {
            // compile network wide burstiness index data in 1s bins
            int idx1 = static_cast<int>( pSpikes[i] * sim_info->deltaT );
            burstinessHist[idx1] = burstinessHist[idx1] + 1.0;

            // compile network wide spike count in 10ms bins
            int idx2 = static_cast<int>( pSpikes[i] * sim_info->deltaT * 100 );
            spikesHistory[idx2] = spikesHistory[idx2] + 1.0;
        }

        // record firing rate to history matrix
        ratesHistory(sim_info->currentStep, iNeuron) = rates[iNeuron];

        // Cap minimum radius size and record radii to history matrix
        // TODO: find out why we cap this here.
        if (radii[iNeuron] < sim_info->minRadius)
            radii[iNeuron] = sim_info->minRadius;

        // record radius to history matrix
        radiiHistory(sim_info->currentStep, iNeuron) = radii[iNeuron];

        DEBUG_MID(cout << "radii[" << iNeuron << ":" << radii[iNeuron] << "]" << endl;)
    }
}

/**
 * Save current simulation state to XML
 * @param[in] sim_info       Pointer to the simulation information.
 * @param[in] neuronTypes	Neuron types: INH or EXC
 * @param[in] starterNeurons	Starter neurons matrix
 * @param[in] neuronThresh	Neuron thresold
 **/
void XmlRecorder::saveSimState(const SimulationInfo* sim_info, VectorMatrix& neuronTypes, VectorMatrix& starterNeurons, VectorMatrix& neuronThresh)
{
    // neuron locations matrices
    VectorMatrix xloc("complete", "const", 1, sim_info->totalNeurons);
    VectorMatrix yloc("complete", "const", 1, sim_info->totalNeurons);

    // Initialize neurons
    for (int i = 0; i < sim_info->totalNeurons; i++)
    {
        xloc[i] = i % sim_info->width;
        yloc[i] = i / sim_info->width;
    }

    // Write XML header information:
    stateOut << "<?xml version=\"1.0\" standalone=\"no\"?>\n" << "<!-- State output file for the DCT growth modeling-->\n";
    //stateOut << version; TODO: version

    // Write the core state information:
    stateOut << "<SimState>\n";
    stateOut << "   " << radiiHistory.toXML("radiiHistory") << endl;
    stateOut << "   " << ratesHistory.toXML("ratesHistory") << endl;
    stateOut << "   " << burstinessHist.toXML("burstinessHist") << endl;
    stateOut << "   " << spikesHistory.toXML("spikesHistory") << endl;
    stateOut << "   " << xloc.toXML("xloc") << endl;
    stateOut << "   " << yloc.toXML("yloc") << endl;
    stateOut << "   " << neuronTypes.toXML("neuronTypes") << endl;

    if (starterNeurons.Size() > 0)
    {
        stateOut << "   " << starterNeurons.toXML("starterNeurons") << endl;
    }

    // Write neuron thresold
    stateOut << "   " << neuronThresh.toXML("neuronThresh") << endl;

    // write time between growth cycles
    stateOut << "   <Matrix name=\"Tsim\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    stateOut << "   " << sim_info->epochDuration << endl;
    stateOut << "</Matrix>" << endl;

    // write simulation end time
    stateOut << "   <Matrix name=\"simulationEndTime\" type=\"complete\" rows=\"1\" columns=\"1\" multiplier=\"1.0\">" << endl;
    stateOut << "   " << g_simulationStep * sim_info->deltaT << endl;
    stateOut << "</Matrix>" << endl;
    stateOut << "</SimState>" << endl;
}

