/**
 *  @file Network.cpp
 *
 *  @author Allan Ortiz 
 *  @author Cory Mayberry
 *
 *  @brief A grid of LIF Neurons and their interconnecting synapses.
 */
#include "Network.h"
#include "AllDSSynapses.h"

/** 
 *  The constructor for Network.
 */
Network::Network(IModel *model, SimulationInfo *simInfo, IRecorder* simRecorder) :
    m_model(model),
    m_sim_info(simInfo),
    m_simRecorder(simRecorder)
{
    cout << "Neuron count: " << simInfo->totalNeurons << endl;
    g_simulationStep = 0;
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
void Network::setup(ISInput* pInput)
{
    cout << "Initializing neurons in network." << endl;
    m_model->setupSim(m_sim_info, m_simRecorder);

    // init stimulus input object
    if (pInput != NULL)
        pInput->init(m_model, *(m_model->getNeurons()), m_sim_info);
}

/**
 *  Begin terminating the simulator.
 */
void Network::finish()
{
    // Terminate the simulator
    m_model->cleanupSim(m_sim_info); // Can #term be removed w/ the new model architecture?  // =>ISIMULATION
}

/**
 *  Advance the network one step in the current epoch. This means do
 *  everything that needs to be done within the model to move
 *  everything forward one time step.
 */
void Network::advance(ISInput* pInput)
{
    // input stimulus
    if (pInput != NULL)
        pInput->inputStimulus(m_model, m_sim_info, m_sim_info->pSummationMap);

    m_model->advance(m_sim_info);
}

/**
 *  Performs growth in the network: updating connections between neurons for the current epoch.
 *  @params currentStep the current epoch of the simulation.
 */
void Network::updateConnections(const int currentStep)
{
    m_model->updateConnections(currentStep, m_sim_info, m_simRecorder);
}

/**
 *  Print debug output of current internal state of the network for the current step.
 */
void Network::logSimStep() const
{
    m_model->logSimStep(m_sim_info);
}

/**
 *  Clean up objects.
 */
void Network::freeResources()
{
}

/**
 * Resets all of the maps.
 * Releases and re-allocates memory for each map, clearing them as necessary.
 */
void Network::reset()
{
    DEBUG(cout << "\nEntering Network::reset()" << endl;)

    // Terminate the simulator
    m_model->cleanupSim(m_sim_info);

    // Clean up objects
    freeResources();

    // Reset global simulation Step to 0
    g_simulationStep = 0;

    // Initialize and prepare network for simulation 
    m_model->setupSim(m_sim_info, m_simRecorder);

    DEBUG(cout << "\nExiting Network::reset()" << endl;)
}

/**
 * Save current simulation state to XML.
 */
void Network::saveState()
{
    m_model->saveState(m_simRecorder);
}

/**
 *  Write the simulation memory image.
 *  @param simulation_step should be an integer type.
 *  @param  os  yhe filestream to write.
 *  This needs to be debugged and verified to be working.
 */
void Network::writeSimMemory(ostream& os)
{
    cerr << "Network::writeSimMemory was called. " << endl;
    // get history matrices with current values
    m_simRecorder->getValues();

    m_model->saveMemory(os, m_sim_info);
}

/**
 *  Read the simulation memory image.
 *  @param  is  the filestream to read.
 */
void Network::readSimMemory(istream& is)
{
    // read the neuron data
    is >> m_sim_info->totalNeurons; is.ignore();
    m_model->loadMemory(is, m_sim_info);

    // Init history matrices with current values
    m_simRecorder->initValues();
}
