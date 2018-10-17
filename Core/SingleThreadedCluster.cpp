#include "SingleThreadedCluster.h"
#include "ISInput.h"

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
    // TODO: we will have the way to specify seed value of each cluster separately.
    clr_info->normRand = new Norm(0, 1, clr_info->seed + clr_info->clusterID); 

    // Create a random number generator used in stimulus input (Poisson)
    clr_info->rng = new MTRand(clr_info->seed + clr_info->clusterID);
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

    // delete a random number generator used in stimulus input (Poisson)
    delete clr_info->rng;

    Cluster::cleanupCluster(sim_info, clr_info);
}

/*
 *  Generates random numbers.
 *
 *  @param  sim_info    SimulationInfo to refer.
 *  @param  clr_info    ClusterInfo to refer.
 */
void SingleThreadedCluster::genRandNumbers(const SimulationInfo *sim_info, ClusterInfo *clr_info)
{
}

/*
 * Advances neurons network state of the cluster one simulation step.
 *
 * @param sim_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 * @param clr_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 * @param iStepOffset - offset from the current simulation step.
 */
void SingleThreadedCluster::advanceNeurons(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset)
{
    m_neurons->advanceNeurons(*m_synapses, sim_info, m_synapseIndexMap, clr_info, iStepOffset);
}

/*
 * Process outgoing spiking data between clusters.
 *
 * @param  clr_info  ClusterInfo to refer.
 */
void SingleThreadedCluster::processInterClustesOutgoingSpikes(ClusterInfo *clr_info)
{
}

/*
 * Process incoming spiking data between clusters.
 *
 * @param  clr_info  ClusterInfo to refer.
 */
void SingleThreadedCluster::processInterClustesIncomingSpikes(ClusterInfo *clr_info)
{
}

/*
 * Advances synapses network state of the cluster one simulation step.
 *
 * @param sim_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 * @param clr_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 * @param iStepOffset - offset from the current simulation step.
 */
void SingleThreadedCluster::advanceSynapses(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset)
{
    m_synapses->advanceSynapses(sim_info, m_neurons, m_synapseIndexMap, iStepOffset);
}

/*
 * Advances synapses spike event queue state of the cluster.
 *
 * @param sim_info - parameters defining the simulation to be run with 
 *                   the given collection of neurons.
 * @param clr_info - parameters defining the simulation to be run with
 *                   the given collection of neurons.
 * @param iStep    - simulation steps to advance.
 */
void SingleThreadedCluster::advanceSpikeQueue(const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStep)
{
    (dynamic_cast<AllSpikingSynapses*>(m_synapses))->advanceSpikeQueue(iStep);

    if (sim_info->pInput != NULL) {
        // advance input stimulus state
        sim_info->pInput->advanceSInputState(clr_info, iStep);
    }
}
