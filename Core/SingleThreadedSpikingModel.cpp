#include "SingleThreadedSpikingModel.h"
#include "AllDSSynapses.h"

/*
 *  Constructor
 */
SingleThreadedSpikingModel::SingleThreadedSpikingModel(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout) : 
    Model(conns, neurons, synapses, layout)
{
}

/*
 * Destructor
 */
SingleThreadedSpikingModel::~SingleThreadedSpikingModel() 
{
	//Let Model base class handle de-allocation
}

/*
 *  Sets up the Simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void SingleThreadedSpikingModel::setupSim(SimulationInfo *sim_info)
{
    Model::setupSim(sim_info);

    // Create a normalized random number generator
    rgNormrnd.push_back(new Norm(0, 1, sim_info->seed));
}

/*
 *  Advance everything in the model one time step. In this case, that
 *  means advancing just the Neurons and Synapses.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void SingleThreadedSpikingModel::advance(const SimulationInfo *sim_info)
{
    m_neurons->advanceNeurons(*m_synapses, sim_info, m_synapseIndexMap);
    m_synapses->advanceSynapses(sim_info, m_neurons, m_synapseIndexMap);
}

/*
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void SingleThreadedSpikingModel::updateConnections(const SimulationInfo *sim_info)
{
    // Update Connections data
    if (m_conns->updateConnections(*m_neurons, sim_info, m_layout)) {
        m_conns->updateSynapsesWeights(sim_info->totalNeurons, *m_neurons, *m_synapses, sim_info, m_layout);
        // create synapse inverse map
        m_synapses->createSynapseImap( m_synapseIndexMap, sim_info );
    }
}

/* -----------------
 * # Helper Functions
 * ------------------
 */

