#include "SingleThreadedCluster.h"

/*
 *  Constructor
 */
SingleThreadedCluster::SingleThreadedCluster(IAllNeurons *neurons, IAllSynapses *synapses) :
    Cluster(neurons, synapses)
{
}

/*
 *  Destructor
 */
SingleThreadedCluster::~SingleThreadedCluster()
{
}

/*
 *  Creates all the Neurons and generates data for them.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      A class to define neurons' layout information in the network.
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void SingleThreadedCluster::setupCluster(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info)
{
    Cluster::setupCluster(sim_info, layout, clr_info);

    // Create a normalized random number generator
    rgNormrnd.push_back(new Norm(0, 1, sim_info->seed)); 
}

/*
 * Advances network state one simulation step.
 *
 * @param sim_info - parameters defining the simulation to be run with the given collection of neurons.
 * @param clr_info - parameters defining the simulation to be run with the given collection of neurons.
 */
void SingleThreadedCluster::advance(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
    m_neurons->advanceNeurons(*m_synapses, sim_info, m_synapseIndexMap, clr_info);
    m_synapses->advanceSynapses(sim_info, m_neurons);
}

/*
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      A class to define neurons' layout information in the network.
 *  @param  conns       A class to define neurons' connections information in the network.
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void SingleThreadedCluster::updateConnections(const SimulationInfo *sim_info, Connections *conns, Layout *layout, const ClusterInfo *clr_info)
{
    // Update Connections data
    if (conns->updateConnections(*m_neurons, sim_info, layout)) {
        conns->updateSynapsesWeights(clr_info->totalClusterNeurons, *m_neurons, *m_synapses, sim_info, layout);
        // create synapse inverse map
        m_synapses->createSynapseImap( m_synapseIndexMap, sim_info, clr_info );
    }
}

