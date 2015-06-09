#include "AllSpikingSynapses.h"

AllSpikingSynapses::AllSpikingSynapses() : AllSynapses()
{
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    tau = NULL;
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
    }

    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    tau = NULL;

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

/**
 *  Reset time varying state vars and recompute decay.
 *  @param  iSyn   index of the synapse to set.
 *  @param  deltaT          inner simulation step duration
 */
void AllSpikingSynapses::resetSynapse(const uint32_t iSyn, const BGFLOAT deltaT)
{
    AllSynapses::resetSynapse(iSyn, deltaT);

    assert( updateDecay(iSyn, deltaT) );
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

/**
 *  Advance one specific Synapse.
 *  @param  iSyn   index of the Synapse to connect to.
 *  @param  deltaT   inner simulation step duration
 */
void AllSpikingSynapses::advanceSynapse(const uint32_t iSyn, const SimulationInfo *sim_info, AllNeurons * neurons)
{
    BGFLOAT &decay = this->decay[iSyn];
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &summationPoint = *(this->summationPoint[iSyn]);

    // is an input in the queue?
    if (isSpikeQueue(iSyn)) {
        changePSR(iSyn, sim_info->deltaT);
    }

    // decay the post spike response
    psr *= decay;
    // and apply it to the summation point
#ifdef USE_OMP
#pragma omp atomic #endif
#endif
    summationPoint += psr;
#ifdef USE_OMP
    //PAB: atomic above has implied flush (following statement generates error -- can't be member variable)
    //#pragma omp flush (summationPoint)
#endif
}

/**
 *  Create a Synapse and connect it to the model.
 *  @param  synapses    the Neuron list to reference.
 *  @param  iSyn   TODO
 *  @param  source  coordinates of the source Neuron.
 *  @param  dest    coordinates of the destination Neuron.
 *  @param  sum_point   TODO
 *  @param  deltaT  TODO
 *  @param  type    type of the Synapse to create.
 */
void AllSpikingSynapses::createSynapse(const uint32_t iSyn, Coordinate source, Coordinate dest, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;

    in_use[iSyn] = true;
    summationPoint[iSyn] = sum_point;
    summationCoord[iSyn] = dest;
    synapseCoord[iSyn] = source;
    W[iSyn] = 10.0e-9;
    this->type[iSyn] = type;
    tau[iSyn] = DEFAULT_tau;

    BGFLOAT tau;
    switch (type) {
        case II:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            assert( false );
            break;
    }

    this->tau[iSyn] = tau;
    this->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    // initializes the queues for the Synapses
    initSpikeQueue(iSyn);
    // reset time varying state vars and recompute decay
    resetSynapse(iSyn, deltaT);
}

void AllSpikingSynapses::changePSR(const uint32_t iSyn, const BGFLOAT deltaT)
{
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &W = this->W[iSyn];
    BGFLOAT &decay = this->decay[iSyn];

    psr += ( W / decay );    // calculate psr
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
