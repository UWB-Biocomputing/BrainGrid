#include "AllSynapsesPolyFuncs.h"

__device__ enumClassSynapses classSynapses_d = undefClassSynapses; 

/* ------------------*\
|* # Device Functions 
\* ------------------*/

/*
 *  Update PSR (post synapse response)
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct
 *                             on device memory.
 *  @param  iSyn               Index of the synapse to set.
 *  @param  simulationStep     The current simulation step.
 *  @param  deltaT             Inner simulation step duration.
 */
__device__ void changeSpikingSynapsesPSRDevice(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, const BGSIZE iSyn, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    BGFLOAT &psr = allSynapsesDevice->psr[iSyn];
    BGFLOAT &W = allSynapsesDevice->W[iSyn];
    BGFLOAT &decay = allSynapsesDevice->decay[iSyn];

    psr += ( W / decay );    // calculate psr
}

/*
 *  Update PSR (post synapse response)
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct
 *                             on device memory.
 *  @param  iSyn               Index of the synapse to set.
 *  @param  simulationStep     The current simulation step.
 *  @param  deltaT             Inner simulation step duration.
 */
__device__ void changeDSSynapsePSRDevice(AllDSSynapsesDeviceProperties* allSynapsesDevice, const BGSIZE iSyn, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    //assert( iSyn < allSynapsesDevice->maxSynapsesPerNeuron * allSynapsesDevice->count_neurons );

    uint64_t &lastSpike = allSynapsesDevice->lastSpike[iSyn];
    BGFLOAT &r = allSynapsesDevice->r[iSyn];
    BGFLOAT &u = allSynapsesDevice->u[iSyn];
    BGFLOAT D = allSynapsesDevice->D[iSyn];
    BGFLOAT F = allSynapsesDevice->F[iSyn];
    BGFLOAT U = allSynapsesDevice->U[iSyn];
    BGFLOAT W = allSynapsesDevice->W[iSyn];
    BGFLOAT &psr = allSynapsesDevice->psr[iSyn];
    BGFLOAT decay = allSynapsesDevice->decay[iSyn];

    // adjust synapse parameters
    if (lastSpike != ULONG_MAX) {
            BGFLOAT isi = (simulationStep - lastSpike) * deltaT ;
            r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
            u = U + u * ( 1 - U ) * exp( -isi / F );
    }
    psr += ( ( W / decay ) * u * r );// calculate psr
    lastSpike = simulationStep; // record the time of the spike
}

/*     
 *  Checks if there is an input spike in the queue.
 *
 *  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
 *                                   on device memory.
 *  @param[in] iSyn                  Index of the Synapse to check.
 *  @return true if there is an input spike event.
 */
__device__ bool isSpikingSynapsesSpikeQueueDevice(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, BGSIZE iSyn)
{
    uint32_t &delay_queue = allSynapsesDevice->delayQueue[iSyn];
    int &delayIdx = allSynapsesDevice->delayIdx[iSyn];
    int ldelayQueue = allSynapsesDevice->ldelayQueue[iSyn];

    uint32_t delayMask = (0x1 << delayIdx);
    bool isFired = delay_queue & (delayMask);
    delay_queue &= ~(delayMask);
    if ( ++delayIdx >= ldelayQueue ) {
            delayIdx = 0;
    }

    return isFired;
}

/*     
 *  Adjust synapse weight according to the Spike-timing-dependent synaptic modification
 *  induced by natural spike trains
 *
 *  @param  allSynapsesDevice    Pointer to the AllSTDPSynapsesDeviceProperties structures 
 *                               on device memory.
 *  @param  iSyn                 Index of the synapse to set.
 *  @param  delta                Pre/post synaptic spike interval.
 *  @param  epost                Params for the rule given in Froemke and Dan (2002).
 *  @param  epre                 Params for the rule given in Froemke and Dan (2002).
 */
__device__ void stdpLearningDevice(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, const BGSIZE iSyn, double delta, double epost, double epre)
{
    BGFLOAT STDPgap = allSynapsesDevice->STDPgap[iSyn];
    BGFLOAT muneg = allSynapsesDevice->muneg[iSyn];
    BGFLOAT mupos = allSynapsesDevice->mupos[iSyn];
    BGFLOAT tauneg = allSynapsesDevice->tauneg[iSyn];
    BGFLOAT taupos = allSynapsesDevice->taupos[iSyn];
    BGFLOAT Aneg = allSynapsesDevice->Aneg[iSyn];
    BGFLOAT Apos = allSynapsesDevice->Apos[iSyn];
    BGFLOAT Wex = allSynapsesDevice->Wex[iSyn];
    BGFLOAT &W = allSynapsesDevice->W[iSyn];
    BGFLOAT dw;

    if (delta < -STDPgap) {
        // Depression
        dw = pow(W, muneg) * Aneg * exp(delta / tauneg);
    } else if (delta > STDPgap) {
        // Potentiation
        dw = pow(Wex - W, mupos) * Apos * exp(-delta / taupos);
    } else {
        return;
    }

    W += epost * epre * dw;

    // check the sign
    if ((Wex < 0 && W > 0) || (Wex > 0 && W < 0)) W = 0;

    // check for greater Wmax
    if (fabs(W) > fabs(Wex)) W = Wex;

    DEBUG_SYNAPSE(
        printf("AllSTDPSynapses::stdpLearning:\n");
        printf("          iSyn: %d\n", iSyn);
        printf("          delta: %f\n", delta);
        printf("          epre: %f\n", epre);
        printf("          epost: %f\n", epost);
        printf("          dw: %f\n", dw);
        printf("          W: %f\n\n", W);
    );
}

/*
 *  Checks if there is an input spike in the queue.
 *
 *  @param[in] allSynapsesDevice     Pointer to AllSTDPSynapsesDeviceProperties structures 
 *                                   on device memory.
 *  @param[in] iSyn                  Index of the Synapse to check.
 *  @return true if there is an input spike event.
 */
__device__ bool isSTDPSynapseSpikeQueuePostDevice(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, BGSIZE iSyn)
{
    uint32_t &delay_queue = allSynapsesDevice->delayQueuePost[iSyn];
    int &delayIdx = allSynapsesDevice->delayIdxPost[iSyn];
    int ldelayQueue = allSynapsesDevice->ldelayQueuePost[iSyn];

    uint32_t delayMask = (0x1 << delayIdx);
    bool isFired = delay_queue & (delayMask);
    delay_queue &= ~(delayMask);
    if ( ++delayIdx >= ldelayQueue ) {
            delayIdx = 0;
    }

    return isFired;
}

/*
 *  Gets the spike history of the neuron.
 *
 *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory. 
 *  @param  index                  Index of the neuron to get spike history.
 *  @param  offIndex               Offset of the history beffer to get.
 *                                 -1 will return the last spike.
 *  @param  max_spikes             Maximum number of spikes per neuron per epoch.
 *  @return Spike history.
 */
__device__ uint64_t getSTDPSynapseSpikeHistoryDevice(AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int index, int offIndex, int max_spikes)
{
    // offIndex is a minus offset
    int idxSp = (allNeuronsDevice->spikeCount[index] + allNeuronsDevice->spikeCountOffset[index] +  max_spikes + offIndex) % max_spikes;
    return allNeuronsDevice->spike_history[index][idxSp];
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

/*
 *  CUDA code for advancing spiking synapses.
 *  Perform updating synapses for one time step.
 *
 *  @param[in] total_synapse_counts  Number of synapses.
 *  @param  synapseIndexMapDevice    Reference to the SynapseIndexMap on device memory.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
 *                                   on device memory.
 */
__global__ void advanceSpikingSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allSynapsesDevice ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= total_synapse_counts )
                return;

        BGSIZE iSyn = synapseIndexMapDevice->activeSynapseIndex[idx];

        BGFLOAT &psr = allSynapsesDevice->psr[iSyn];
        BGFLOAT decay = allSynapsesDevice->decay[iSyn];

        // Checks if there is an input spike in the queue.
        bool isFired = isSpikingSynapsesSpikeQueueDevice(allSynapsesDevice, iSyn);

        // is an input in the queue?
        if (isFired) {
                switch (classSynapses_d) {
                case classAllSpikingSynapses:
                       changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                        break;
                case classAllDSSynapses:
                        changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                        break;
                default:
                        assert(false);
                }
        }
        // decay the post spike response
        psr *= decay;
}

/*
 *  CUDA code for advancing STDP synapses.
 *  Perform updating synapses for one time step.
 *
 *  @param[in] total_synapse_counts  Number of synapses.
 *  @param  synapseIndexMapDevice    Reference to the SynapseIndexMap on device memory.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] allSynapsesDevice     Pointer to AllSTDPSynapsesDeviceProperties structures 
 *                                   on device memory.
 */
__global__ void advanceSTDPSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapsesDeviceProperties* allSynapsesDevice, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int max_spikes, int width ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= total_synapse_counts )
            return;

    BGSIZE iSyn = synapseIndexMapDevice->activeSynapseIndex[idx];

    BGFLOAT &decay = allSynapsesDevice->decay[iSyn];
    BGFLOAT &psr = allSynapsesDevice->psr[iSyn];

    // is an input in the queue?
    bool fPre = isSpikingSynapsesSpikeQueueDevice(allSynapsesDevice, iSyn);
    bool fPost = isSTDPSynapseSpikeQueuePostDevice(allSynapsesDevice, iSyn);
    if (fPre || fPost) {
        BGFLOAT &tauspre = allSynapsesDevice->tauspre[iSyn];
        BGFLOAT &tauspost = allSynapsesDevice->tauspost[iSyn];
        BGFLOAT &taupos = allSynapsesDevice->taupos[iSyn];
        BGFLOAT &tauneg = allSynapsesDevice->tauneg[iSyn];
        int &total_delay = allSynapsesDevice->total_delay[iSyn];
        bool &useFroemkeDanSTDP = allSynapsesDevice->useFroemkeDanSTDP[iSyn];

        // pre and post neurons index
        int idxPre = allSynapsesDevice->sourceNeuronIndex[iSyn];
        int idxPost = allSynapsesDevice->destNeuronIndex[iSyn];
        int64_t spikeHistory, spikeHistory2;
        BGFLOAT delta;
        BGFLOAT epre, epost;

        if (fPre) {     // preSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time 
            // just one before the last spike.
            spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPre, -2, max_spikes);
            if (spikeHistory > 0 && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = ((int64_t)simulationStep - spikeHistory) * deltaT;
                epre = 1.0 - exp(-delta / tauspre);
            } else {
                epre = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // pre-post spikes
            int offIndex = -1;  // last spike
            while (true) {
                spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPost, offIndex, max_spikes);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between pre-post spikes
                delta = (spikeHistory - (int64_t)simulationStep) * deltaT;

                DEBUG_SYNAPSE(
                    printf("advanceSTDPSynapsesDevice: fPre\n");
                    printf("          iSyn: %d\n", iSyn);
                    printf("          idxPre: %d\n", idxPre);
                    printf("          idxPost: %d\n", idxPost);
                    printf("          spikeHistory: %d\n", spikeHistory);
                    printf("          simulationStep: %d\n", simulationStep);
                    printf("          delta: %f\n\n", delta);
                );

                if (delta <= -3.0 * tauneg)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPost, offIndex-1, max_spikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epost = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspost);
                } else {
                    epost = 1.0;
                }
                stdpLearningDevice(allSynapsesDevice, iSyn, delta, epost, epre);
                --offIndex;
            }

            switch (classSynapses_d) {
            case classAllSTDPSynapses:
                changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                break;
            case classAllDynamicSTDPSynapses:
                // Note: we cast void * over the allSynapsesDevice, then recast it, 
                // because AllDSSynapsesDeviceProperties inherited properties from 
                // the AllDSSynapsesDeviceProperties and the AllSTDPSynapsesDeviceProperties.
                changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>((void *)allSynapsesDevice), iSyn, simulationStep, deltaT);
                break;
            default:
                assert(false);
            }
        }

        if (fPost) {    // postSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time
            // just one before the last spike.
            spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPost, -2, max_spikes);
            if (spikeHistory > 0 && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = ((int64_t)simulationStep - spikeHistory) * deltaT;
                epost = 1.0 - exp(-delta / tauspost);
            } else {
                epost = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // post-pre spikes
            int offIndex = -1;  // last spike
            while (true) {
                spikeHistory = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPre, offIndex, max_spikes);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between post-pre spikes
                delta = ((int64_t)simulationStep - spikeHistory - total_delay) * deltaT;

                DEBUG_SYNAPSE(
                    printf("advanceSTDPSynapsesDevice: fPost\n");
                    printf("          iSyn: %d\n", iSyn);
                    printf("          idxPre: %d\n", idxPre);
                    printf("          idxPost: %d\n", idxPost);
                    printf("          spikeHistory: %d\n", spikeHistory);
                    printf("          simulationStep: %d\n", simulationStep);
                    printf("          delta: %f\n\n", delta);
                );

                if (delta <= 0 || delta >= 3.0 * taupos)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPre, offIndex-1, max_spikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epre = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspre);
                } else {
                    epre = 1.0;
                }
                stdpLearningDevice(allSynapsesDevice, iSyn, delta, epost, epre);
                --offIndex;
            }
        }
    }

    // decay the post spike response
    psr *= decay;
}

