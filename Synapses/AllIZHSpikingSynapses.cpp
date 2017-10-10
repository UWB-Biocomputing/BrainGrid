// Updated 5/15 by Jewel
// look for "IZH03" for modifications

#include "AllIZHSpikingSynapses.h"

AllIZHSpikingSynapses::AllIZHSpikingSynapses() : AllSpikingSynapses()
{
    decay = NULL;
    total_delay = NULL;
    delayQueue = NULL;
    delayIdx = NULL;
    ldelayQueue = NULL;
    tau = NULL;
}

AllIZHSpikingSynapses::~AllIZHSpikingSynapses()
{
    cleanupSynapses();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
/*
void AllIZHSpikingSynapses::setupSynapses(SimulationInfo *sim_info)
{
    setupSynapses(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
}
*/

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */ 
void AllIZHSpikingSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    AllSynapses::resetSynapse(iSyn, deltaT);
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
void AllIZHSpikingSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    in_use[iSyn] = true;
    summationPoint[iSyn] = sum_point;
    destNeuronIndex[iSyn] = dest_index;
    sourceNeuronIndex[iSyn] = source_index;
    W[iSyn] = synSign(type) * 10.0e-9;

// IZH03:
//    this->type[iSyn] = type;
// tau value = default (tau is not used when we don't have decay factor) 
    tau[iSyn] = DEFAULT_tau;
// IZH03:
// make delay the same as one step size deltaT
// unit of total_delay is simulation time step not second
// and delay is defined in second, so we convert second into time step.
    BGFLOAT delay = 5e-4; // delay = deltaT
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
bool AllIZHSpikingSynapses::isSpikeQueue(const BGSIZE iSyn)
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
void AllIZHSpikingSynapses::preSpikeHit(const BGSIZE iSyn)
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
void AllIZHSpikingSynapses::postSpikeHit(const BGSIZE iSyn)
{
}

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 */
void AllIZHSpikingSynapses::advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons * neurons)
{
    BGFLOAT &decay = this->decay[iSyn];
    BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &summationPoint = *(this->summationPoint[iSyn]);
// IZH03: since there is no decay, before the next time step, psr = 0 
	psr = 0;
    // is an input in the queue?
    if (isSpikeQueue(iSyn)) {
        changePSR(iSyn, sim_info->deltaT);
    }
// IZH03: no decay 
    //psr *= decay; // decay the post spike response and apply it to the summation point
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
void AllIZHSpikingSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT)
{
	BGFLOAT &psr = this->psr[iSyn];
    BGFLOAT &W = this->W[iSyn];
// IZH03: no decay in izh model, only synaptic weights contribute to psr
    //BGFLOAT &decay = this->decay[iSyn];
    //psr += ( W / decay );    // calculate psr
    psr += W ;    // calculate psr
}

#endif //!defined(USE_GPU)
