#include "SingleThreadedSpikingModel.h"
#include "AllDSSynapses.h"

/*
*  Constructor
*/
SingleThreadedSpikingModel::SingleThreadedSpikingModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 
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

/**
 *  Advance everything in the model one time step. In this case, that
 *  means advancing just the Neurons and Synapses.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void SingleThreadedSpikingModel::advance(const SimulationInfo *sim_info)
{
    m_neurons->advanceNeurons(*m_synapses, sim_info, m_synapseIndexMap);
    m_synapses->advanceSynapses(sim_info, m_neurons);
}

/**
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void SingleThreadedSpikingModel::updateConnections(const SimulationInfo *sim_info)
{
    // Update Connections data
    if (m_conns->updateConnections(*m_neurons, sim_info)) {
        m_conns->updateSynapsesWeights(sim_info->totalNeurons, *m_neurons, *m_synapses, sim_info);
        // create synapse inverse map
        createSynapseImap( *m_synapses, sim_info );
    }
}

/* -----------------
* # Helper Functions
* ------------------
*/

