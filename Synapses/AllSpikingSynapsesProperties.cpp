#include "AllSpikingSynapsesProperties.h"
#include "EventQueue.h"

// Default constructor
AllSpikingSynapsesProperties::AllSpikingSynapsesProperties() : AllSynapsesProperties()
{
    decay = NULL;
    total_delay = NULL;
    tau = NULL;
    preSpikeQueue = NULL;
}

AllSpikingSynapsesProperties::~AllSpikingSynapsesProperties()
{
    cleanupSynapsesProperties();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingSynapsesProperties::setupSynapsesProperties(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSynapsesProperties::setupSynapsesProperties(num_neurons, max_synapses, sim_info, clr_info);

    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        decay = new BGFLOAT[max_total_synapses];
        total_delay = new int[max_total_synapses];
        tau = new BGFLOAT[max_total_synapses];

        // create a pre synapse spike queue & initialize it
        preSpikeQueue = new EventQueue();
#if defined(USE_GPU)
        // max_total_synapses * sim_info->maxFiringRate may overflow the maximum range
        // of 32 bits integer, so we cast it uint64_t
        int nMaxInterClustersOutgoingEvents = (uint64_t) max_total_synapses * sim_info->maxFiringRate * sim_info->deltaT * sim_info->minSynapticTransDelay;
        int nMaxInterClustersIncomingEvents = (uint64_t) max_total_synapses * sim_info->maxFiringRate * sim_info->deltaT * sim_info->minSynapticTransDelay;

        // initializes the pre synapse spike queue
        preSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses, nMaxInterClustersOutgoingEvents, nMaxInterClustersIncomingEvents);
#else // USE_GPU
        // initializes the pre synapse spike queue
        preSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses);
#endif // USE_GPU

        // register the queue to the event handler
        if (clr_info->eventHandler != NULL) {
            clr_info->eventHandler->addEventQueue(clr_info->clusterID, preSpikeQueue);
        }
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllSpikingSynapsesProperties::cleanupSynapsesProperties()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] decay;
        delete[] total_delay;
        delete[] tau;
    }

    decay = NULL;
    total_delay = NULL;
    tau = NULL;

    if (preSpikeQueue != NULL) {
        delete preSpikeQueue;
        preSpikeQueue = NULL;
    }

    AllSynapsesProperties::cleanupSynapsesProperties();
}
