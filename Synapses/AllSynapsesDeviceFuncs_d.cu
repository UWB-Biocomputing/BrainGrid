#include "AllSynapsesDeviceFuncs.h"
#include "AllSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "math_constants.h"
#include "HeapSort.hpp"


// a device variable to store synapse class ID.
__device__ enumClassSynapses classSynapses_d = undefClassSynapses;

/* --------------------------------------*\
|* # Device Functions for advanceSynapses
\* --------------------------------------*/

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

    psr += (W / decay);    // calculate psr
}

/*
 *  Update PSR (post synapse response)
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct
 *                             on device memory.
 *  @param  iSyn               Index of the synapse to set.
 *  @param  simulationStep     The current simulation step.
 *  @param  deltaT             Inner simulation step duration.
 *  @param  iStepOffset        Offset from the current simulation step.
 */
__device__ void changeDSSynapsePSRDevice(AllDSSynapsesDeviceProperties* allSynapsesDevice, const BGSIZE iSyn, const uint64_t simulationStep, const BGFLOAT deltaT, int iStepOffset)
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
        BGFLOAT isi = (simulationStep + iStepOffset - lastSpike) * deltaT;
        r = 1 + (r * (1 - u) - 1) * exp(-isi / D);
        u = U + u * (1 - U) * exp(-isi / F);
    }
    psr += ((W / decay) * u * r);// calculate psr
    lastSpike = simulationStep + iStepOffset; // record the time of the spike
}

/*
 *  Checks if there is an input spike in the queue.
 *
 *  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures
 *                                   on device memory.
 *  @param[in] iSyn                  Index of the Synapse to check.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 *  @return true if there is an input spike event.
 */
__device__ bool isSpikingSynapsesSpikeQueueDevice(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, BGSIZE iSyn, int iStepOffset)
{
    int &total_delay = allSynapsesDevice->total_delay[iSyn];

    // Checks if there is an event in the queue.
    return allSynapsesDevice->preSpikeQueue->checkAnEvent(iSyn, total_delay, iStepOffset);
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
    }
    else if (delta > STDPgap) {
        // Potentiation
        dw = pow(Wex - W, mupos) * Apos * exp(-delta / taupos);
    }
    else {
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
 *  @param[in] iStepOffset           Offset from the current simulation step.
 *  @return true if there is an input spike event.
 */
__device__ bool isSTDPSynapseSpikeQueuePostDevice(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, BGSIZE iSyn, int iStepOffset)
{
    return allSynapsesDevice->postSpikeQueue->checkAnEvent(iSyn, iStepOffset);
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
    int idxSp = (allNeuronsDevice->spikeCount[index] + allNeuronsDevice->spikeCountOffset[index] + max_spikes + offIndex) % max_spikes;
    return allNeuronsDevice->spike_history[index][idxSp];
}

/* --------------------------------------*\
|* # Global Functions for advanceSynapses
\* --------------------------------------*/

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
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__global__ void advanceSpikingSynapsesDevice(int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, int iStepOffset, int threadGranularity) {
    int idx_raw = 4* (blockIdx.x * blockDim.x + threadIdx.x);

    for (int i = 0; i < 4; i++) {
        int idx = idx_raw + i;
        if (idx >= total_synapse_counts)
            return;
        BGSIZE iSyn = synapseIndexMapDevice->incomingSynapseIndexMap[idx];

        // is an input in the queue?
        if (isSpikingSynapsesSpikeQueueDevice(allSynapsesDevice, iSyn, iStepOffset)) {
            switch (classSynapses_d) {
            case classAllSpikingSynapses:
                changeSpikingSynapsesPSRDevice(static_cast<AllSpikingSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT);
                break;
            case classAllDSSynapses:
                changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>(allSynapsesDevice), iSyn, simulationStep, deltaT, iStepOffset);
                break;
            default:
                assert(false);
            }
        }

        // decay the post spike response
        allSynapsesDevice->psr[iSyn] *= allSynapsesDevice->decay[iSyn];
    }
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
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__global__ void advanceSTDPSynapsesDevice(int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapsesDeviceProperties* allSynapsesDevice, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int max_spikes, int width, int iStepOffset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_synapse_counts)
        return;

    BGSIZE iSyn = synapseIndexMapDevice->incomingSynapseIndexMap[idx];

    BGFLOAT &decay = allSynapsesDevice->decay[iSyn];
    BGFLOAT &psr = allSynapsesDevice->psr[iSyn];

    // is an input in the queue?
    bool fPre = isSpikingSynapsesSpikeQueueDevice(allSynapsesDevice, iSyn, iStepOffset);
    bool fPost = isSTDPSynapseSpikeQueuePostDevice(allSynapsesDevice, iSyn, iStepOffset);
    if (fPre || fPost) {
        BGFLOAT &tauspre = allSynapsesDevice->tauspre[iSyn];
        BGFLOAT &tauspost = allSynapsesDevice->tauspost[iSyn];
        BGFLOAT &taupos = allSynapsesDevice->taupos[iSyn];
        BGFLOAT &tauneg = allSynapsesDevice->tauneg[iSyn];
        int &total_delay = allSynapsesDevice->total_delay[iSyn];
        bool &useFroemkeDanSTDP = allSynapsesDevice->useFroemkeDanSTDP[iSyn];

        // pre and post neurons index
        int idxPre = allSynapsesDevice->sourceNeuronLayoutIndex[iSyn];
        int idxPost = allSynapsesDevice->destNeuronLayoutIndex[iSyn];
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
                delta = ((int64_t)simulationStep + iStepOffset - spikeHistory) * deltaT;
                epre = 1.0 - exp(-delta / tauspre);
            }
            else {
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
                delta = (spikeHistory - ((int64_t)simulationStep + iStepOffset)) * deltaT;

                DEBUG_SYNAPSE(
                    printf("advanceSTDPSynapsesDevice: fPre\n");
                printf("          iSyn: %d\n", iSyn);
                printf("          idxPre: %d\n", idxPre);
                printf("          idxPost: %d\n", idxPost);
                printf("          spikeHistory: %d\n", spikeHistory);
                printf("          simulationStep: %d\n", simulationStep + iStepOffset);
                printf("          delta: %f\n\n", delta);
                );

                if (delta <= -3.0 * tauneg)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPost, offIndex - 1, max_spikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epost = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspost);
                }
                else {
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
                changeDSSynapsePSRDevice(static_cast<AllDSSynapsesDeviceProperties*>((void *)allSynapsesDevice), iSyn, simulationStep, deltaT, iStepOffset);
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
                delta = ((int64_t)simulationStep + iStepOffset - spikeHistory) * deltaT;
                epost = 1.0 - exp(-delta / tauspost);
            }
            else {
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
                delta = ((int64_t)simulationStep + iStepOffset - spikeHistory - total_delay) * deltaT;

                DEBUG_SYNAPSE(
                    printf("advanceSTDPSynapsesDevice: fPost\n");
                printf("          iSyn: %d\n", iSyn);
                printf("          idxPre: %d\n", idxPre);
                printf("          idxPost: %d\n", idxPost);
                printf("          spikeHistory: %d\n", spikeHistory);
                printf("          simulationStep: %d\n", simulationStep + iStepOffset);
                printf("          delta: %f\n\n", delta);
                );

                if (delta <= 0 || delta >= 3.0 * taupos)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = getSTDPSynapseSpikeHistoryDevice(allNeuronsDevice, idxPre, offIndex - 1, max_spikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epre = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspre);
                }
                else {
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

/* ------------------------------------*\
|* # Device Functions for createSynapse
\* ------------------------------------*/

/*
 * Return 1 if originating neuron is excitatory, -1 otherwise.
 *
 * @param[in] t  synapseType I to I, I to E, E to I, or E to E
 * @return 1 or -1
 */
__device__ int synSign(synapseType t)
{
    switch (t)
    {
    case II:
    case IE:
        return -1;
    case EI:
    case EE:
        return 1;
    }

    return 0;
}

/*
 *  Create a Spiking Synapse and connect it to the model.
 *
 *  @param allSynapsesDevice    Pointer to the AllSpikingSynapsesDeviceProperties structures
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createSpikingSynapse(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesDevice->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesDevice->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->psr[iSyn] = 0.0;
    allSynapsesDevice->type[iSyn] = type;

    allSynapsesDevice->tau[iSyn] = DEFAULT_tau;

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
        break;
    }

    allSynapsesDevice->tau[iSyn] = tau;
    allSynapsesDevice->decay[iSyn] = exp(-deltaT / tau);
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>(delay / deltaT) + 1;

    assert(allSynapsesDevice->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY);

    // initializes the queues for the Synapses
    allSynapsesDevice->preSpikeQueue->clearAnEvent(iSyn);
}

/*
 *  Create a DS Synapse and connect it to the model.
 *
 *  @param allSynapsesDevice    Pointer to the AllSpikingSynapsesDeviceProperties structures
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createDSSynapse(AllDSSynapsesDeviceProperties* allSynapsesDevice, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesDevice->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesDevice->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->psr[iSyn] = 0.0;
    allSynapsesDevice->r[iSyn] = 1.0;
    allSynapsesDevice->u[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesDevice->lastSpike[iSyn] = ULONG_MAX;
    allSynapsesDevice->type[iSyn] = type;

    allSynapsesDevice->U[iSyn] = DEFAULT_U;
    allSynapsesDevice->tau[iSyn] = DEFAULT_tau;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    BGFLOAT tau;
    switch (type) {
    case II:
        U = 0.32;
        D = 0.144;
        F = 0.06;
        tau = 6e-3;
        delay = 0.8e-3;
        break;
    case IE:
        U = 0.25;
        D = 0.7;
        F = 0.02;
        tau = 6e-3;
        delay = 0.8e-3;
        break;
    case EI:
        U = 0.05;
        D = 0.125;
        F = 1.2;
        tau = 3e-3;
        delay = 0.8e-3;
        break;
    case EE:
        U = 0.5;
        D = 1.1;
        F = 0.05;
        tau = 3e-3;
        delay = 1.5e-3;
        break;
    default:
        break;
    }

    allSynapsesDevice->U[iSyn] = U;
    allSynapsesDevice->D[iSyn] = D;
    allSynapsesDevice->F[iSyn] = F;

    allSynapsesDevice->tau[iSyn] = tau;
    allSynapsesDevice->decay[iSyn] = exp(-deltaT / tau);
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>(delay / deltaT) + 1;

    assert(allSynapsesDevice->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY);

    // initializes the queues for the Synapses
    allSynapsesDevice->preSpikeQueue->clearAnEvent(iSyn);
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param allSynapsesDevice    Pointer to the AllSpikingSynapsesDeviceProperties structures
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createSTDPSynapse(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesDevice->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesDevice->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->psr[iSyn] = 0.0;
    allSynapsesDevice->type[iSyn] = type;

    allSynapsesDevice->tau[iSyn] = DEFAULT_tau;

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
        break;
    }

    allSynapsesDevice->tau[iSyn] = tau;
    allSynapsesDevice->decay[iSyn] = exp(-deltaT / tau);
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>(delay / deltaT) + 1;

    assert(allSynapsesDevice->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY);

    allSynapsesDevice->Apos[iSyn] = 0.5;
    allSynapsesDevice->Aneg[iSyn] = -0.5;
    allSynapsesDevice->STDPgap[iSyn] = 2e-3;

    allSynapsesDevice->total_delayPost[iSyn] = 0;

    allSynapsesDevice->tauspost[iSyn] = 0;
    allSynapsesDevice->tauspre[iSyn] = 0;

    allSynapsesDevice->taupos[iSyn] = 15e-3;
    allSynapsesDevice->tauneg[iSyn] = 35e-3;
    allSynapsesDevice->Wex[iSyn] = 1.0;

    allSynapsesDevice->mupos[iSyn] = 0;
    allSynapsesDevice->muneg[iSyn] = 0;

    allSynapsesDevice->useFroemkeDanSTDP[iSyn] = false;

    // initializes the queues for the Synapses
    allSynapsesDevice->postSpikeQueue->clearAnEvent(iSyn);
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param allSynapsesDevice    Pointer to the AllSpikingSynapsesDeviceProperties structures
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createDynamicSTDPSynapse(AllDynamicSTDPSynapsesDeviceProperties* allSynapsesDevice, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesDevice->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesDevice->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->psr[iSyn] = 0.0;
    allSynapsesDevice->r[iSyn] = 1.0;
    allSynapsesDevice->u[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesDevice->lastSpike[iSyn] = ULONG_MAX;
    allSynapsesDevice->type[iSyn] = type;

    allSynapsesDevice->U[iSyn] = DEFAULT_U;
    allSynapsesDevice->tau[iSyn] = DEFAULT_tau;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    BGFLOAT tau;
    switch (type) {
    case II:
        U = 0.32;
        D = 0.144;
        F = 0.06;
        tau = 6e-3;
        delay = 0.8e-3;
        break;
    case IE:
        U = 0.25;
        D = 0.7;
        F = 0.02;
        tau = 6e-3;
        delay = 0.8e-3;
        break;
    case EI:
        U = 0.05;
        D = 0.125;
        F = 1.2;
        tau = 3e-3;
        delay = 0.8e-3;
        break;
    case EE:
        U = 0.5;
        D = 1.1;
        F = 0.05;
        tau = 3e-3;
        delay = 1.5e-3;
        break;
    default:
        break;
    }

    allSynapsesDevice->U[iSyn] = U;
    allSynapsesDevice->D[iSyn] = D;
    allSynapsesDevice->F[iSyn] = F;

    allSynapsesDevice->tau[iSyn] = tau;
    allSynapsesDevice->decay[iSyn] = exp(-deltaT / tau);
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>(delay / deltaT) + 1;

    assert(allSynapsesDevice->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY);

    allSynapsesDevice->Apos[iSyn] = 0.5;
    allSynapsesDevice->Aneg[iSyn] = -0.5;
    allSynapsesDevice->STDPgap[iSyn] = 2e-3;

    allSynapsesDevice->total_delayPost[iSyn] = 0;

    allSynapsesDevice->tauspost[iSyn] = 0;
    allSynapsesDevice->tauspre[iSyn] = 0;

    allSynapsesDevice->taupos[iSyn] = 15e-3;
    allSynapsesDevice->tauneg[iSyn] = 35e-3;
    allSynapsesDevice->Wex[iSyn] = 1.0;

    allSynapsesDevice->mupos[iSyn] = 0;
    allSynapsesDevice->muneg[iSyn] = 0;

    allSynapsesDevice->useFroemkeDanSTDP[iSyn] = false;

    // initializes the queues for the Synapses
    allSynapsesDevice->postSpikeQueue->clearAnEvent(iSyn);
}

/*
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 *
 * @param allSynapsesDevice      Pointer to the AllSpikingSynapsesDeviceProperties structures
 *                               on device memory.
 * @param type                   Type of the Synapse to create.
 * @param neuron_index           Index of the destination neuron in the cluster.
 * @param source_index           Layout index of the source neuron.
 * @param dest_index             Layout index of the destination neuron.
 * @param sum_point              Pointer to the summation point.
 * @param deltaT                 The time step size.
 * @param weight                 Synapse weight.
 */
__device__ void addSpikingSynapse(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, synapseType type, const int neuron_index, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT weight)
{
    if (allSynapsesDevice->synapse_counts[neuron_index] >= allSynapsesDevice->maxSynapsesPerNeuron) {
        assert(false);
        return; // TODO: ERROR!
    }

    // add it to the list
    BGSIZE synapse_index;
    BGSIZE max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    BGSIZE synapseBegin = max_synapses * neuron_index;
    for (synapse_index = 0; synapse_index < max_synapses; synapse_index++) {
        if (!allSynapsesDevice->in_use[synapseBegin + synapse_index]) {
            break;
        }
    }

    allSynapsesDevice->synapse_counts[neuron_index]++;

    // create a synapse
    switch (classSynapses_d) {
    case classAllSpikingSynapses:
        createSpikingSynapse(allSynapsesDevice, neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type);
        break;
    case classAllDSSynapses:
        createDSSynapse(static_cast<AllDSSynapsesDeviceProperties *>(allSynapsesDevice), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type);
        break;
    case classAllSTDPSynapses:
        createSTDPSynapse(static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type);
        break;
    case classAllDynamicSTDPSynapses:
        createDynamicSTDPSynapse(static_cast<AllDynamicSTDPSynapsesDeviceProperties *>(allSynapsesDevice), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type);
        break;
    default:
        assert(false);
    }
    allSynapsesDevice->W[synapseBegin + synapse_index] = weight;
}

/*
 * Remove a synapse from the network.
 *
 * @param[in] allSynapsesDevice      Pointer to the AllSpikingSynapsesDeviceProperties structures
 *                                   on device memory.
 * @param neuron_index               Index of the destination neuron in the cluster.
 * @param synapse_offset             Offset into neuron_index's synapses.
 * @param[in] maxSynapses            Maximum number of synapses per neuron.
 */
__device__ void eraseSpikingSynapse(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, const int neuron_index, const int synapse_offset, int maxSynapses)
{
    BGSIZE iSync = maxSynapses * neuron_index + synapse_offset;
    allSynapsesDevice->synapse_counts[neuron_index]--;
    allSynapsesDevice->in_use[iSync] = false;
}

/*
 * Returns the type of synapse at the given coordinates
 *
 * @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
 * @param src_neuron             Index of the source neuron.
 * @param dest_neuron            Index of the destination neuron.
 */
__device__ synapseType synType(neuronType* neuron_type_map_d, const int src_neuron, const int dest_neuron)
{
    if (neuron_type_map_d[src_neuron] == INH && neuron_type_map_d[dest_neuron] == INH)
        return II;
    else if (neuron_type_map_d[src_neuron] == INH && neuron_type_map_d[dest_neuron] == EXC)
        return IE;
    else if (neuron_type_map_d[src_neuron] == EXC && neuron_type_map_d[dest_neuron] == INH)
        return EI;
    else if (neuron_type_map_d[src_neuron] == EXC && neuron_type_map_d[dest_neuron] == EXC)
        return EE;

    return STYPE_UNDEF;

}

/* -------------------------------------*\
|* # Global Functions for updateSynapses
\* -------------------------------------*/

/*
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below
 * zero.
 *
 * @param[in] num_neurons        Total number of neurons.
 * @param[in] deltaT             The time step size.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
 * @param[in] neuron_type_map_d   Pointer to the neurons type map in device memory.
 * @param[in] totalClusterNeurons  Total number of neurons in the cluster.
 * @param[in] clusterNeuronsBegin  Begin neuron index of the cluster.
 * @param[in] radii_d            Pointer to the rates data array.
 * @param[in] xloc_d             Pointer to the neuron's x location array.
 * @param[in] yloc_d             Pointer to the neuron's y location array.
 */
__global__ void updateSynapsesWeightsDevice(int num_neurons, BGFLOAT deltaT, int maxSynapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, neuronType* neuron_type_map_d, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* radii_d, BGFLOAT* xloc_d, BGFLOAT* yloc_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalClusterNeurons)
        return;

    int adjusted = 0;
    int removed = 0;
    int added = 0;

    int iNeuron = idx;
    int dest_neuron = clusterNeuronsBegin + idx;

    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        if (dest_neuron == src_neuron) {
            // we don't create a synapse between the same neuron.
            continue;
        }

        // Update the areas of overlap in between Neurons
        BGFLOAT distX = xloc_d[dest_neuron] - xloc_d[src_neuron];
        BGFLOAT distY = yloc_d[dest_neuron] - yloc_d[src_neuron];
        BGFLOAT dist2 = distX * distX + distY * distY;
        BGFLOAT dist = sqrt(dist2);
        BGFLOAT delta = dist - (radii_d[dest_neuron] + radii_d[src_neuron]);
        BGFLOAT area = 0.0;

        if (delta < 0) {
            BGFLOAT lenAB = dist;
            BGFLOAT r1 = radii_d[dest_neuron];
            BGFLOAT r2 = radii_d[src_neuron];

            if (lenAB + min(r1, r2) <= max(r1, r2)) {
                area = CUDART_PI_F * min(r1, r2) * min(r1, r2); // Completely overlapping unit
            }
            else {
                // Partially overlapping unit
                BGFLOAT lenAB2 = dist2;
                BGFLOAT r12 = r1 * r1;
                BGFLOAT r22 = r2 * r2;

                BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                BGFLOAT angCBA = acos(cosCBA);
                BGFLOAT angCBD = 2.0 * angCBA;

                BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                BGFLOAT angCAB = acos(cosCAB);
                BGFLOAT angCAD = 2.0 * angCAB;

                area = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
            }
        }

        // visit each synapse at (xa,ya)
        bool connected = false;
        synapseType type = synType(neuron_type_map_d, src_neuron, dest_neuron);

        // for each existing synapse
        BGSIZE existing_synapses = allSynapsesDevice->synapse_counts[iNeuron];
        int existing_synapses_checked = 0;
        for (BGSIZE synapse_index = 0; (existing_synapses_checked < existing_synapses) && !connected; synapse_index++) {
            BGSIZE iSyn = maxSynapses * iNeuron + synapse_index;
            if (allSynapsesDevice->in_use[iSyn] == true) {
                // if there is a synapse between a and b
                if (allSynapsesDevice->sourceNeuronLayoutIndex[iSyn] == src_neuron) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.

                    // W_d[] is indexed by (dest_neuron (local index) * totalNeurons + src_neuron)
                    if (area < 0) {
                        removed++;
                        eraseSpikingSynapse(allSynapsesDevice, iNeuron, synapse_index, maxSynapses);
                    }
                    else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesDevice->W[iSyn] = area * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
                    }
                }
                existing_synapses_checked++;
            }
        }

        // if not connected and weight(a,b) > 0, add a new synapse from a to b
        if (!connected && (area > 0)) {
            // locate summation point
            BGFLOAT* sum_point = &(allNeuronsDevice->summation_map[iNeuron]);
            added++;

            BGFLOAT weight = area * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
            addSpikingSynapse(allSynapsesDevice, type, iNeuron, src_neuron, dest_neuron, sum_point, deltaT, weight);
        }
    }
}

/*
 *  CUDA kernel function for setting up connections.
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters:
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param  num_neurons         Number of total neurons.
 *  @param  totalClusterNeurons	Total number of neurons in the cluster.
 *  @param  clusterNeuronsBegin Begin neuron index of the cluster.
 *  @param  xloc_d              Pointer to the neuron's x location array.
 *  @param  yloc_d              Pointer to the neuron's y location array.
 *  @param  nConnsPerNeuron     Number of maximum connections per neurons.
 *  @param  threshConnsRadius   Connection radius threshold.
 *  @param  neuron_type_map_d   Pointer to the neurons type map in device memory.
 *  @param  rDistDestNeuron_d   Pointer to the DistDestNeuron structure array.
 *  @param  deltaT              The time step size.
 *  @param  allNeuronsDevice    Pointer to the Neuron structures in device memory.
 *  @param  allSynapsesDevice   Pointer to the Synapse structures in device memory.
 *  @param  minExcWeight        Min values of excitatory neuron's synapse weight.
 *  @param  maxExcWeight        Max values of excitatory neuron's synapse weight.
 *  @param  minInhWeight        Min values of inhibitory neuron's synapse weight.
 *  @param  maxInhWeight        Max values of inhibitory neuron's synapse weight.
 *  @param  devStates_d         Curand global state.
 *  @param  seed                Seed for curand.
 */
__global__ void setupConnectionsDevice(int num_neurons, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* xloc_d, BGFLOAT* yloc_d, int nConnsPerNeuron, int threshConnsRadius, neuronType* neuron_type_map_d, ConnStatic::DistDestNeuron *rDistDestNeuron_d, BGFLOAT deltaT, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, BGFLOAT minExcWeight, BGFLOAT maxExcWeight, BGFLOAT minInhWeight, BGFLOAT maxInhWeight, curandState* devStates_d, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalClusterNeurons)
        return;

    int iNeuron = idx;
    int dest_neuron = iNeuron + clusterNeuronsBegin;

    // pick the connections shorter than threshConnsRadius
    BGSIZE iArrayBegin = num_neurons * iNeuron, iArrayEnd, iArray;
    iArray = iArrayBegin;
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        if (src_neuron != dest_neuron) {
            BGFLOAT distX = xloc_d[dest_neuron] - xloc_d[src_neuron];
            BGFLOAT distY = yloc_d[dest_neuron] - yloc_d[src_neuron];
            BGFLOAT dist2 = distX * distX + distY * distY;
            BGFLOAT dist = sqrt(dist2);

            if (dist <= threshConnsRadius) {
                ConnStatic::DistDestNeuron distDestNeuron;
                distDestNeuron.dist = dist;
                distDestNeuron.src_neuron = src_neuron;
                rDistDestNeuron_d[iArray++] = distDestNeuron;
            }
        }
    }


    // sort ascendant
    iArrayEnd = iArray;
    int size = iArrayEnd - iArrayBegin;
    // CUDA thrust sort consumes heap memory, and when sorting large contents
    // it may cause an error "temporary_buffer::allocate: get_temporary_buffer failed".
    // Therefore we use local implementation of heap sort.
    // NOTE: Heap sort is an in-palce algoprithm (memory requirement is 1).
    // Its implementation is not stable. Time complexity is O(n*logn).
    heapSort(&rDistDestNeuron_d[iArrayBegin], size);

    // set up an initial state for curand
    curand_init(seed, iNeuron, 0, &devStates_d[iNeuron]);

    // pick the shortest nConnsPerNeuron connections
    iArray = iArrayBegin;
    for (BGSIZE i = 0; iArray < iArrayEnd && (int)i < nConnsPerNeuron; iArray++, i++) {
        ConnStatic::DistDestNeuron distDestNeuron = rDistDestNeuron_d[iArray];
        int src_neuron = distDestNeuron.src_neuron;
        synapseType type = synType(neuron_type_map_d, src_neuron, dest_neuron);

        // create a synapse at the cluster of the destination neuron

        DEBUG_MID(printf("source: %d dest: %d dist: %d\n", src_neuron, dest_neuron, distDestNeuron.dist); )

            // set synapse weight
            // TODO: we need another synaptic weight distibution mode (normal distribution)
            BGFLOAT weight;
        curandState localState = devStates_d[iNeuron];
        if (synSign(type) > 0) {
            weight = minExcWeight + curand_uniform(&localState) * (maxExcWeight - minExcWeight);
        }
        else {
            weight = minInhWeight + curand_uniform(&localState) * (maxInhWeight - minInhWeight);
        }
        devStates_d[iNeuron] = localState;

        BGFLOAT* sum_point = &(allNeuronsDevice->summation_map[iNeuron]);
        addSpikingSynapse(allSynapsesDevice, type, iNeuron, src_neuron, dest_neuron, sum_point, deltaT, weight);

    }
}

/*
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 *
 * @param allSynapsesDevice      Pointer to the Synapse structures in device memory.
 * @param pSummationMap          Pointer to the summation point.
 * @param width                  Width of neuron map (assumes square).
 * @param deltaT                 The simulation time step size.
 * @param weight                 Synapse weight.
 */
__global__ void initSynapsesDevice(int n, AllDSSynapsesDeviceProperties* allSynapsesDevice, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // create a synapse
    int neuron_index = idx;
    BGFLOAT* sum_point = &(pSummationMap[neuron_index]);
    synapseType type = allSynapsesDevice->type[neuron_index];
    createDSSynapse(allSynapsesDevice, neuron_index, 0, 0, neuron_index, sum_point, deltaT, type);
    allSynapsesDevice->W[neuron_index] = weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
}

/*
 * Perform updating preSpikeQueue for one time step.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct
 *                             on device memory.
 *  @param  iStep              Simulation steps to advance.
 */
__global__ void advanceSpikingSynapsesEventQueueDevice(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, int iStep)
{
    allSynapsesDevice->preSpikeQueue->advanceEventQueue(iStep);
}

/*
 * Perform updating postSpikeQueue for one time step.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct
 *                             on device memory.
 *  @param  iStep              Simulation steps to advance.
 */
__global__ void advanceSTDPSynapsesEventQueueDevice(AllSTDPSynapsesDeviceProperties* allSynapsesDevice, int iStep)
{
    allSynapsesDevice->postSpikeQueue->advanceEventQueue(iStep);
}

