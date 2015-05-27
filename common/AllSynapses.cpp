#include "AllSynapses.h"

AllSynapses::AllSynapses() :
        maxSynapsesPerNeuron(0),
        total_synapse_counts(0),
        count_neurons(0)
{
    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;
}

AllSynapses::AllSynapses(const int num_neurons, const int max_synapses) 
{
    setupSynapses(num_neurons, max_synapses);
}

AllSynapses::~AllSynapses()
{
    cleanupSynapses();
}

void AllSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

void AllSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    uint32_t max_total_synapses = max_synapses * num_neurons;

    maxSynapsesPerNeuron = max_synapses;
    total_synapse_counts = 0;
    count_neurons = num_neurons;

    if (max_total_synapses != 0) {
        summationCoord = new Coordinate[max_total_synapses];
        W = new BGFLOAT[max_total_synapses];
        summationPoint = new BGFLOAT*[max_total_synapses];
        synapseCoord = new Coordinate[max_total_synapses];
        psr = new BGFLOAT[max_total_synapses];
        type = new synapseType[max_total_synapses];
        in_use = new bool[max_total_synapses];
        synapse_counts = new size_t[num_neurons];

        for (int i = 0; i < max_total_synapses; i++) {
            summationPoint[i] = NULL;
            in_use[i] = false;
        }

        for (int i = 0; i < num_neurons; i++) {
            synapse_counts[i] = 0;
        }
    }
}

void AllSynapses::cleanupSynapses()
{
    uint32_t max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] summationCoord;
        delete[] W;
        delete[] summationPoint;
        delete[] synapseCoord;
        delete[] psr;
        delete[] type;
        delete[] in_use;
        delete[] synapse_counts;
    }

    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    type = NULL;
    in_use = NULL;
    synapse_counts = NULL;

    count_neurons = 0;
    maxSynapsesPerNeuron = 0;
}

#if !defined(USE_GPU)
/**
 *  Advance all the Synapses in the simulation.
 *  param  sim_info    SimulationInfo class to read information from.
 */
void AllSynapses::advanceSynapses(const SimulationInfo *sim_info)
{
    int num_neurons = sim_info->totalNeurons;
    BGFLOAT deltaT = sim_info->deltaT;

    for (int i = 0; i < num_neurons; i++) {
        size_t synapse_counts = this->synapse_counts[i];
        int synapse_advanced = 0;
        for (int z = 0; z < synapse_counts; z++) {
            // Advance Synapse
            uint32_t iSyn = maxSynapsesPerNeuron * i + z;
            advanceSynapse(iSyn, deltaT);
            synapse_advanced++;
        }
    }
}

/**
 *  Remove a synapse from the network.
 *  @param  neuron_index   Index of a neuron.
 *  @param  iSyn      Index of a synapse.
 */
void AllSynapses::eraseSynapse(const int neuron_index, const uint32_t iSyn)
{
    synapse_counts[neuron_index]--;
    in_use[iSyn] = false;
    summationPoint[iSyn] = NULL;
}

/**
 *  Adds a Synapse to the model, connecting two Neurons.
 *  @param  type    the type of the Synapse to add.
 *  @param  type    the weight of the Synapse.
 *  @param  src_neuron  the Neuron that sends to this Synapse.
 *  @param  dest_neuron the Neuron that receives from the Synapse.
 *  @param  source  coordinates of the source Neuron.
 *  @param  dest    coordinates of the destination Neuron.
 *  @param  sum_point   TODO
 *  @param deltaT   inner simulation step duration
 */
void AllSynapses::addSynapse(BGFLOAT weight, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, BGFLOAT *sum_point, const BGFLOAT deltaT)
{
    if (synapse_counts[src_neuron] >= maxSynapsesPerNeuron) {
        return; // TODO: ERROR!
    }

    // add it to the list
    size_t synapse_index;
    uint32_t iSyn;
    for (synapse_index = 0; synapse_index < maxSynapsesPerNeuron; synapse_index++) {
        iSyn = maxSynapsesPerNeuron * src_neuron + synapse_index;
        if (!in_use[iSyn]) {
            break;
        }
    }

    synapse_counts[src_neuron]++;

    // create a synapse
    createSynapse(iSyn, source, dest, sum_point, deltaT, type );

    W[iSyn] = weight;
}
#endif
