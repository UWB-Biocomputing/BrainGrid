#include "AllSynapsesProperties.h"

// Default constructor
AllSynapsesProperties::AllSynapsesProperties()
{
    destNeuronLayoutIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronLayoutIndex = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;
}

AllSynapsesProperties::~AllSynapsesProperties()
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
void AllSynapsesProperties::setupSynapsesProperties(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    count_neurons = num_neurons;
    maxSynapsesPerNeuron = max_synapses;
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    total_synapse_counts = 0;

    if (max_total_synapses != 0) {
        destNeuronLayoutIndex = new int[max_total_synapses];
        W = new BGFLOAT[max_total_synapses];
        summationPoint = new BGFLOAT*[max_total_synapses];
        sourceNeuronLayoutIndex = new int[max_total_synapses];
        psr = new BGFLOAT[max_total_synapses];
        type = new synapseType[max_total_synapses];
        in_use = new bool[max_total_synapses];
        synapse_counts = new BGSIZE[num_neurons];

        for (BGSIZE i = 0; i < max_total_synapses; i++) {
            summationPoint[i] = NULL;
            in_use[i] = false;
        }

        for (int i = 0; i < num_neurons; i++) {
            synapse_counts[i] = 0;
        }
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllSynapsesProperties::cleanupSynapsesProperties()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] destNeuronLayoutIndex;
        delete[] W;
        delete[] summationPoint;
        delete[] sourceNeuronLayoutIndex;
        delete[] psr;
        delete[] type;
        delete[] in_use;
        delete[] synapse_counts;
    }

    destNeuronLayoutIndex = NULL;
    W = NULL;
    summationPoint = NULL;
    sourceNeuronLayoutIndex = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;

    count_neurons = 0;
    maxSynapsesPerNeuron = 0;
}
