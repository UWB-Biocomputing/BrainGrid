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
 *  @param  growthEpochDuration  the duration of each growth in the simulation.
 *  @param  maxGrowthSteps  the maximum amount of steps for this simulation.
 */
void Network::setup(BGFLOAT growthEpochDuration, BGFLOAT maxGrowthSteps)
{

    m_model->setupSim(neurons.size, m_sim_info);
}

/**
 *  Begin terminating the simulator.
 *  @param  growthEpochDuration    the duration of each growth in the simulation.
 *  @param  maxGrowthSteps  the maximum amount of steps for this simulation.
 */
void Network::finish(BGFLOAT growthEpochDuration, BGFLOAT maxGrowthSteps)
{
    // Terminate the simulator
    m_model->cleanupSim(neurons, m_sim_info); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

/**
 *  Advance the network one step in the current epoch. This means do
 *  everything that needs to be done within the model to move
 *  everything forward one time step.
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
    m_model->updateConnections(currentStep, neurons, synapses, m_sim_info);
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
 *  @param simulation_step should be an integer type.
 *  @param  os  yhe filestream to write.
 *  This needs to be debugged and verified to be working.
 */
void Network::writeSimMemory(BGFLOAT simulation_step, ostream& os)
{
	cerr << "Network::writeSimMemory was called. " << endl;
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
