#include "AllSpikingSynapses.h"

AllSpikingSynapses::AllSpikingSynapses() : AllSynapses()
{
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    tau = NULL;
    lastSpike = NULL;
}

AllSpikingSynapses::AllSpikingSynapses(const int num_neurons, const int max_synapses) 
{
    setupSynapses(num_neurons, max_synapses);
}

AllSpikingSynapses::~AllSpikingSynapses()
{
    cleanupSynapses();
}

void AllSpikingSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

void AllSpikingSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    AllSynapses::setupSynapses(num_neurons, max_synapses);

    uint32_t max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        decay = new BGFLOAT[max_total_synapses];
        total_delay = new int[max_total_synapses];
        delayQueue = new uint32_t[max_total_synapses];
        delayIdx = new int[max_total_synapses];
        ldelayQueue = new int[max_total_synapses];
        tau = new BGFLOAT[max_total_synapses];
        lastSpike = new uint64_t[max_total_synapses];
    }
}

void AllSpikingSynapses::cleanupSynapses()
{
    uint32_t max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] decay;
        delete[] total_delay;
        delete[] delayQueue;
        delete[] delayIdx;
        delete[] ldelayQueue;
        delete[] tau;
        delete[] lastSpike;
    }

    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    tau = NULL;
    lastSpike = NULL;

    AllSynapses::cleanupSynapses();
}

/**
 *  Initializes the queues for the Synapses.
 *  @param  iSyn   index of the synapse to set.
 */
void AllSpikingSynapses::initSpikeQueue(const uint32_t iSyn)
{
    int &total_delay = this->total_delay[iSyn];
    uint32_t &delayQueue = this->delayQueue[iSyn];
    int &delayIdx = this->delayIdx[iSyn];
    int &ldelayQueue = this->ldelayQueue[iSyn];

    size_t size = total_delay / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
    delayQueue = 0;
    delayIdx = 0;
    ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

#if !defined(USE_GPU)
/**
 *  Prepares Synapse for a spike hit.
 *  @param  iSyn   index of the Synapse to update.
 */
void AllSpikingSynapses::preSpikeHit(const uint32_t iSyn)
{
    uint32_t &delay_queue = this->delayQueue[iSyn];
    int &delayIdx = this->delayIdx[iSyn];
    int &ldelayQueue = this->ldelayQueue[iSyn];
    int &total_delay = this->total_delay[iSyn];

    // Add to spike queue

    // calculate index where to insert the spike into delayQueue
    int idx = delayIdx +  total_delay;
    if ( idx >= ldelayQueue ) {
        idx -= ldelayQueue;
    }

    // set a spike
    assert( !(delay_queue & (0x1 << idx)) );
    delay_queue |= (0x1 << idx);
}

bool AllSpikingSynapses::allowBackPropagation()
{
    return false;
}

void AllSpikingSynapses::postSpikeHit(const uint32_t iSyn)
{
}
#endif
