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

/*
 *  Sets the data for Synapse #synapse_index from Neuron #neuron_index.
 *  @param  input   istream to read from.
 *  @param  iSyn   index of the synapse to set.
 */
void AllSpikingSynapses::readSynapse(istream &input, const uint32_t iSyn)
{
    AllSynapses::readSynapse(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> decay[iSyn]; input.ignore();
    input >> total_delay[iSyn]; input.ignore();
    input >> delayQueue[iSyn]; input.ignore();
    input >> delayIdx[iSyn]; input.ignore();
    input >> ldelayQueue[iSyn]; input.ignore();
    input >> tau[iSyn]; input.ignore();
    input >> lastSpike[iSyn]; input.ignore();
}

/**
 *  Write the synapse data to the stream.
 *  @param  output  stream to print out to.
 *  @param  iSyn   index of the synapse to print out.
 */
void AllSpikingSynapses::writeSynapse(ostream& output, const uint32_t iSyn) const
{
    AllSynapses::writeSynapse(output, iSyn);

    output << decay[iSyn] << ends;
    output << total_delay[iSyn] << ends;
    output << delayQueue[iSyn] << ends;
    output << delayIdx[iSyn] << ends;
    output << ldelayQueue[iSyn] << ends;
    output << tau[iSyn] << ends;
    output << lastSpike[iSyn] << ends;
}

#if !defined(USE_GPU)
/**
 *  Checks if there is an input spike in the queue.
 *  @param  iSyn   index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool AllSpikingSynapses::isSpikeQueue(const uint32_t iSyn)
{
    uint32_t &delayQueue = this->delayQueue[iSyn];
    int &delayIdx = this->delayIdx[iSyn];
    int &ldelayQueue = this->ldelayQueue[iSyn];

    bool r = delayQueue & (0x1 << delayIdx);
    delayQueue &= ~(0x1 << delayIdx);
    if ( ++delayIdx >= ldelayQueue ) {
        delayIdx = 0;
    }
    return r;
}

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

void AllSpikingSynapses::postSpikeHit(const uint32_t iSyn)
{
}

#endif //!defined(USE_GPU)

/**
 *  Updates the decay if the synapse selected.
 *  @param  iSyn   index of the synapse to set.
 *  @param  deltaT  inner simulation step duration
 */
bool AllSpikingSynapses::updateDecay(const uint32_t iSyn, const BGFLOAT deltaT)
{
        BGFLOAT &tau = this->tau[iSyn];
        BGFLOAT &decay = this->decay[iSyn];

        if (tau > 0) {
                decay = exp( -deltaT / tau );
                return true;
        }
        return false;
}

bool AllSpikingSynapses::allowBackPropagation()
{
    return false;
}
