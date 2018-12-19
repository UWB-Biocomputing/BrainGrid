#include "AllSTDPSynapsesProperties.h"
#include "EventQueue.h"

// Default constructor
AllSTDPSynapsesProperties::AllSTDPSynapsesProperties() : AllSpikingSynapsesProperties()
{
    total_delayPost = NULL;
    tauspost = NULL;
    tauspre = NULL;
    taupos = NULL;
    tauneg = NULL;
    STDPgap = NULL;
    Wex = NULL;
    Aneg = NULL;
    Apos = NULL;
    mupos = NULL;
    muneg = NULL;
    useFroemkeDanSTDP = NULL;
    postSpikeQueue = NULL;
}

AllSTDPSynapsesProperties::~AllSTDPSynapsesProperties()
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
void AllSTDPSynapsesProperties::setupSynapsesProperties(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingSynapsesProperties::setupSynapsesProperties(num_neurons, max_synapses, sim_info, clr_info);

    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        total_delayPost = new int[max_total_synapses];
        tauspost = new BGFLOAT[max_total_synapses];
        tauspre = new BGFLOAT[max_total_synapses];
        taupos = new BGFLOAT[max_total_synapses];
        tauneg = new BGFLOAT[max_total_synapses];
        STDPgap = new BGFLOAT[max_total_synapses];
        Wex = new BGFLOAT[max_total_synapses];
        Aneg = new BGFLOAT[max_total_synapses];
        Apos = new BGFLOAT[max_total_synapses];
        mupos = new BGFLOAT[max_total_synapses];
        muneg = new BGFLOAT[max_total_synapses];
        useFroemkeDanSTDP = new bool[max_total_synapses];

        // create a post synapse spike queue & initialize it
        postSpikeQueue = new EventQueue();
#if defined(USE_GPU)
        // initializes the post synapse spike queue
        postSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses, (int)0, (int)0);
#else // USE_GPU
        // initializes the post synapse spike queue
        postSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses);
#endif // USE_GPU
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllSTDPSynapsesProperties::cleanupSynapsesProperties()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] total_delayPost;
        delete[] tauspost;
        delete[] tauspre;
        delete[] taupos;
        delete[] tauneg;
        delete[] STDPgap;
        delete[] Wex;
        delete[] Aneg;
        delete[] Apos;
        delete[] mupos;
        delete[] muneg;
        delete[] useFroemkeDanSTDP;
    }

    total_delayPost = NULL;
    tauspost = NULL;
    tauspre = NULL;
    taupos = NULL;
    tauneg = NULL;
    STDPgap = NULL;
    Wex = NULL;
    Aneg = NULL;
    Apos = NULL;
    mupos = NULL;
    muneg = NULL;
    useFroemkeDanSTDP = NULL;

    if (postSpikeQueue != NULL) {
        delete postSpikeQueue;
        postSpikeQueue = NULL;
    }

    AllSpikingSynapsesProperties::cleanupSynapsesProperties();
}
