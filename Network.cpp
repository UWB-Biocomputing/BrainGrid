/**
 *  @file Network.cpp
 *
 *  @author Allan Ortiz & Cory Mayberry
 *
 *  @brief A grid of LIF Neurons and their interconnecting synapses.
 */
#include "Network.h"

/** 
 * The constructor for Network.
 * @post The network is setup according to parameters and ready for simulation.
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
    m_model->createAllNeurons(neurons, m_sim_info);
}

/**
* Destructor
*
*/
Network::~Network()
{
    freeResources();
}

/**
 * Initialize and prepare network for simulation.
 *
 * @param growthStepDuration
 *
 * @param maxGrowthSteps
 */
void Network::setup(FLOAT growthStepDuration, FLOAT maxGrowthSteps)
{

    m_model->setupSim(neurons.size, m_sim_info);
}

void Network::finish(FLOAT growthStepDuration, FLOAT maxGrowthSteps)
{
    // Terminate the simulator
    m_model->cleanupSim(neurons, m_sim_info); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

void Network::advance()
{
    m_model->advance(neurons, synapses, m_sim_info);
}

void Network::updateConnections(const int currentStep)
{
    m_model->updateConnections(currentStep, neurons, synapses, m_sim_info);
}

void Network::logSimStep() const
{
    m_model->logSimStep(neurons, synapses, m_sim_info);
}

/**
* Clean up heap objects
*
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

    m_summationMap = new FLOAT[m_sim_info.cNeurons];

    // initialize maps
    for (int i = 0; i < m_sim_info.cNeurons; i++)
    {
        m_summationMap[i] = 0;
    }

    m_sim_info.pSummationMap = m_summationMap;

    DEBUG(cout << "\nExiting Network::reset()";)
}

/**
* Save current simulation state to XML
*
* @param os
*/
void Network::saveState(ostream& os)
{
    m_model->saveState(os, neurons, m_sim_info);
}

/**
* Write the simulation memory image
*
* @param os	The filestream to write
*/
void Network::writeSimMemory(FLOAT simulation_step, ostream& os)
{
    m_model->saveMemory(os, neurons, synapses, simulation_step);
}

/**
* Read the simulation memory image
*
* @param is	The filestream to read
*/
void Network::readSimMemory(istream& is)
{
    // read the neuron data
    is >> neurons.size;
    assert(neurons.size == m_sim_info.cNeurons);
    m_model->loadMemory(is, neurons, synapses, m_sim_info);
}
