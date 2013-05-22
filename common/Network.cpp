/**
 *  @file Network.cpp
 *
 *  @author Allan Ortiz 
 *  @author Cory Mayberry
 *
 *  @brief A grid of LIF Neurons and their interconnecting synapses.
 */
#include "Network.h"

/** 
 *  The constructor for Network.
 */
Network::Network(Model *model, SimulationInfo &simInfo) :
    m_model(model),
    neurons(simInfo.cNeurons),
    synapses(simInfo.cNeurons,simInfo.maxSynapsesPerNeuron),
    m_summationMap(NULL),
    m_sim_info(simInfo)
{
    cout << "Neuron count: " << simInfo.cNeurons << endl;
    g_simulationStep = 0;
    cout << "Initializing neurons in network." << endl;
    m_model->createAllNeurons(neurons, m_sim_info);
    setup();
	m_model->initializeModel(m_sim_info, neurons, synapses);
}

/**
 *  Destructor.
 */
Network::~Network()
{
    freeResources();
}

/**
 *  Initialize and prepare network for simulation.
 */
void Network::setup()
{

    m_model->setupSim(neurons.size, m_sim_info);
}

/**
 *  Begin terminating the simulator.
 */
void Network::finish()
{
    // Terminate the simulator
    m_model->cleanupSim(neurons, m_sim_info); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

/**
 *  Advance the network one step in the current epoch.
 */
void Network::advance()
{
    m_model->advance(neurons, synapses, m_sim_info);
}

/**
 *  Performs growth in the network: updating connections between neurons for the current epoch.
 *  @params currentStep the current epoch of the simulation.
 */
void Network::updateConnections(const int currentStep)
{
    // Calculate growth cycle firing rate for previous period
    m_model->getSpikeCounts(neurons);
    m_model->updateHistory(currentStep, m_sim_info.stepDuration, neurons);
    m_model->updateFrontiers(neurons.size);
    m_model->updateOverlap(neurons.size);
    m_model->updateWeights(neurons.size, neurons, synapses, m_sim_info);
    // clear spike count
    m_model->clearSpikeCounts(neurons);
}

/**
 *  Print debug output of current internal state of the network for the current step.
 */
void Network::logSimStep() const
{
    m_model->logSimStep(neurons, synapses, m_sim_info);
}

/**
 *  Clean up heap objects.
 */
void Network::freeResources()
{
    if (m_summationMap != NULL) 
		delete[] m_summationMap;
}

/**
 * Resets all of the maps.
 * Releases and re-allocates memory for each map, clearing them as necessary.
 */
void Network::reset()
{
    DEBUG(cout << "\nEntering Network::reset()";)

    freeResources();

    neurons = AllNeurons(m_sim_info.cNeurons);
    synapses = AllSynapses(m_sim_info.cNeurons, m_sim_info.maxSynapsesPerNeuron);

    // Reset global simulation Step to 0
    g_simulationStep = 0;

    m_summationMap = new BGFLOAT[m_sim_info.cNeurons];

    // initialize maps
    for (int i = 0; i < m_sim_info.cNeurons; i++)
    {
        m_summationMap[i] = 0;
    }

    m_sim_info.pSummationMap = m_summationMap;

    DEBUG(cout << "\nExiting Network::reset()";)
}

/**
 * Save current simulation state to XML.
 * @param   os  the output stream to send the state data to.
 */
void Network::saveState(ostream& os)
{
    m_model->saveState(os, neurons, m_sim_info);
}

/**
 *  Write the simulation memory image.
 *  @param  os  yhe filestream to write.
 */
void Network::writeSimMemory(BGFLOAT simulation_step, ostream& os)
{
    m_model->saveMemory(os, neurons, synapses, simulation_step);
}

/**
 *  Read the simulation memory image.
 *  @param  is  the filestream to read.
 */
void Network::readSimMemory(istream& is)
{
    // read the neuron data
    is >> neurons.size;
    assert(neurons.size == m_sim_info.cNeurons);
    m_model->loadMemory(is, neurons, synapses, m_sim_info);
}
