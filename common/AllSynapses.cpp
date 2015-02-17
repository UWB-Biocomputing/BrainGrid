#include "AllSynapses.h"

AllSynapses::AllSynapses() :
        max_synapses(0),
        total_synapse_counts(0),
        count_neurons(0)
{
    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    type = NULL;
    tau = NULL;
    lastSpike = NULL;
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
    this->max_synapses = max_synapses;

    total_synapse_counts = 0;
    count_neurons = num_neurons;

    summationCoord = new Coordinate*[num_neurons];
    W = new BGFLOAT*[num_neurons];
    summationPoint = new BGFLOAT**[num_neurons];
    synapseCoord = new Coordinate*[num_neurons];
    psr = new BGFLOAT*[num_neurons];
    decay = new BGFLOAT*[num_neurons];
    total_delay = new int*[num_neurons];
    delayQueue = new uint32_t**[num_neurons];
    delayIdx = new int*[num_neurons];
    ldelayQueue = new int*[num_neurons];
    type = new synapseType*[num_neurons];
    tau = new BGFLOAT*[num_neurons];
    lastSpike = new uint64_t*[num_neurons];
    in_use = new bool*[num_neurons];
    synapse_counts = new size_t[num_neurons];

    if (max_synapses != 0) {
        for (int i = 0; i < num_neurons; i++) {
            summationCoord[i] = new Coordinate[max_synapses];
            W[i] = new BGFLOAT[max_synapses];
            summationPoint[i] = new BGFLOAT*[max_synapses];
            synapseCoord[i] = new Coordinate[max_synapses];
            psr[i] = new BGFLOAT[max_synapses];
            decay[i] = new BGFLOAT[max_synapses];
            total_delay[i] = new int[max_synapses];
            delayQueue[i] = new uint32_t*[max_synapses];
            delayIdx[i] = new int[max_synapses];
            ldelayQueue[i] = new int[max_synapses];
            type[i] = new synapseType[max_synapses];
            tau[i] = new BGFLOAT[max_synapses];
            lastSpike[i] = new uint64_t[max_synapses];
            in_use[i] = new bool[max_synapses];

            for (int j = 0; j < max_synapses; j++) {
                summationPoint[i][j] = NULL;
                delayQueue[i][j] = new uint32_t[1];
                delayIdx[i][j] = 0;
                ldelayQueue[i][j] = 0;
                in_use[i][j] = false;
            }

            synapse_counts[i] = 0;
        }
    }
}

void AllSynapses::cleanupSynapses()
{
    if (count_neurons != 0) {
        if (max_synapses != 0) {
            for (int i = 0; i < count_neurons; i++) {
                // Release references to summation points - while held by the synapse, these references are
                // not owned by the synapse.
                for (size_t j = 0; j < max_synapses; j++) {
                    summationPoint[i][j] = NULL;
                    delete [] delayQueue[i][j];
                    delayQueue[i][j] = NULL;
                }

                delete[] summationCoord[i];
                delete[] W[i];
                delete[] summationPoint[i];
                delete[] synapseCoord[i];
                delete[] psr[i];
                delete[] decay[i];
                delete[] total_delay[i];
                delete[] delayQueue[i];
                delete[] delayIdx[i];
                delete[] ldelayQueue[i];
                delete[] type[i];
                delete[] tau[i];
                delete[] lastSpike[i];
                delete[] in_use[i];
            }
        }

        delete[] summationCoord;
        delete[] W;
        delete[] summationPoint;
        delete[] synapseCoord;
        delete[] psr;
        delete[] decay;
        delete[] total_delay;
        delete[] delayQueue;
        delete[] delayIdx;
        delete[] ldelayQueue;
        delete[] type;
        delete[] tau;
        delete[] lastSpike;
        delete[] in_use;
        delete[] synapse_counts;
    }

    summationCoord = NULL;
    W = NULL;
    summationPoint = NULL;
    synapseCoord = NULL;
    psr = NULL;
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    type = NULL;
    tau = NULL;
    lastSpike = NULL;
    in_use = NULL;
    synapse_counts = NULL;

    count_neurons = 0;
    max_synapses = 0;
}

/**
 *  Initializes the queues for the Synapses.
 *  @param  neuron_index    index of the neuron that the synapse belongs to.
 *  @param  synapse_index   index of the synapse to set.
 */
void AllSynapses::initSpikeQueue(const int neuron_index, const int synapse_index)
{
        int &total_delay = this->total_delay[neuron_index][synapse_index];
        uint32_t &delayQueue = this->delayQueue[neuron_index][synapse_index][0];
        int &delayIdx = this->delayIdx[neuron_index][synapse_index];
        int &ldelayQueue = this->ldelayQueue[neuron_index][synapse_index];

        size_t size = total_delay / ( sizeof(uint8_t) * 8 ) + 1;
        assert( size <= BYTES_OF_DELAYQUEUE );
        delayQueue = 0;
        delayIdx = 0;
        ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

