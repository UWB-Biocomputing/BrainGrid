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
    clr_info->normRand = new Norm(0, 1, clr_info->seed); 
}

/*
 *  Clean up the cluster.
 *
 *  @param  sim_info    SimulationInfo to refer.
 *  @param  clr_info    ClusterInfo to refer.
 */
void SingleThreadedCluster::cleanupCluster(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    // delete a normalized random number generator
    delete clr_info->normRand;

    Cluster::cleanupCluster(sim_info, clr_info);
}

/*
 * Advances neurons network state of the cluster one simulation step.
 *
 * @param sim_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 * @param clr_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 */
void SingleThreadedCluster::advanceNeurons(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
    m_neurons->advanceNeurons(*m_synapses, sim_info, m_synapseIndexMap, clr_info);
}

/*
 * Advances synapses network state of the cluster one simulation step.
 *
 * @param sim_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 * @param clr_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 */
void SingleThreadedCluster::advanceSynapses(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
    m_synapses->advanceSynapses(sim_info, m_neurons, m_synapseIndexMap);
}

/*
 * Advances synapses pre spike event queue state of the cluster one simulation step.
 */
void SingleThreadedCluster::advancePreSpikeQueue()
{
    m_synapses->advancePreSpikeQueue();
}
