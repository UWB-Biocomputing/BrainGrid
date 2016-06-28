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

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSpikingSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::setupSynapses(const int num_neurons, const int max_synapses)
{
    AllSynapses::setupSynapses(num_neurons, max_synapses);

    BGSIZE max_total_synapses = max_synapses * num_neurons;

    if (max_total_synapses != 0) {
        decay = new BGFLOAT[max_total_synapses];
        total_delay = new int[max_total_synapses];
        delayQueue = new uint32_t[max_total_synapses];
        delayIdx = new int[max_total_synapses];
        ldelayQueue = new int[max_total_synapses];
        tau = new BGFLOAT[max_total_synapses];
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSpikingSynapses::cleanupSynapses()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

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

/*
 *  Initializes the queues for the Synapse.
 *
 *  @param  iSyn   index of the synapse to set.
 */
void AllSpikingSynapses::initSpikeQueue(const BGSIZE iSyn)
{
    int &total_delay = this->total_delay[iSyn];
    uint32_t &delayQueue = this->delayQueue[iSyn];
    int &delayIdx = this->delayIdx[iSyn];
    int &ldelayQueue = this->ldelayQueue[iSyn];

    uint32_t size = total_delay / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
    delayQueue = 0;
    delayIdx = 0;
    ldelayQueue = LENGTH_OF_DELAYQUEUE;
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */
void AllSpikingSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    AllSynapses::resetSynapse(iSyn, deltaT);

    assert( updateDecay(iSyn, deltaT) );
}

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllSpikingSynapses::checkNumParameters()
{
    return (nParams >= 0);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllSpikingSynapses::readParameters(const TiXmlElement& element)
{
    return false;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllSpikingSynapses::printParameters(ostream &output) const
{
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSpikingSynapses::readSynapse(istream &input, const BGSIZE iSyn)
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

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSpikingSynapses::writeSynapse(ostream& output, const BGSIZE iSyn) const
{
    AllSynapses::writeSynapse(output, iSyn);

    output << decay[iSyn] << ends;
    output << total_delay[iSyn] << ends;
    output << delayQueue[iSyn] << ends;
    output << delayIdx[iSyn] << ends;
    output << ldelayQueue[iSyn] << ends;
    output << tau[iSyn] << ends;
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param  synapses    The synapse list to reference.
 *  @param  iSyn        Index of the synapse to set.
 *  @param  source      Coordinates of the source Neuron.
 *  @param  dest        Coordinates of the destination Neuron.
 *  @param  sum_point   Summation point address.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  type        Type of the Synapse to create.
 */
void AllSpikingSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;

    in_use[iSyn] = true;
    summationPoint[iSyn] = sum_point;
    destNeuronIndex[iSyn] = dest_index;
    sourceNeuronIndex[iSyn] = source_index;
    W[iSyn] = synSign(type) * 10.0e-9;
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

#if !defined(USE_GPU)
/*
 *  Checks if there is an input spike in the queue.
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @return true if there is an input spike event.
 */
bool AllSpikingSynapses::isSpikeQueue(const BGSIZE iSyn)
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

/*
 *  Prepares Synapse for a spike hit.
 *
 *  @param  iSyn   Index of the Synapse to update.
 */
void AllSpikingSynapses::preSpikeHit(const BGSIZE iSyn)
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

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to update.
 */
void AllSpikingSynapses::postSpikeHit(const BGSIZE iSyn)
{
}

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 */
void AllSpikingSynapses::advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons * neurons)
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

/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  deltaT      Inner simulation step duration.
 */
void AllSpikingSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &W = this->W[iSyn];
    BGFLOAT &decay = this->decay[iSyn];

    psr += ( W / decay );    // calculate psr
}

#endif //!defined(USE_GPU)

/*
 *  Updates the decay if the synapse selected.
 *
 *  @param  iSyn    Index of the synapse to set.
 *  @param  deltaT  Inner simulation step duration
 */
bool AllSpikingSynapses::updateDecay(const BGSIZE iSyn, const BGFLOAT deltaT)
{
        BGFLOAT &tau = this->tau[iSyn];
        BGFLOAT &decay = this->decay[iSyn];

        if (tau > 0) {
                decay = exp( -deltaT / tau );
                return true;
        }
        return false;
}

/*
 *  Check if the back propagation (notify a spike event to the pre neuron)
 *  is allowed in the synapse class.
 *
 *  @retrun true if the back propagation is allowed.
 */
bool AllSpikingSynapses::allowBackPropagation()
{
    return false;
}
