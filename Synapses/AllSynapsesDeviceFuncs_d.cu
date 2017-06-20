#include "AllSynapsesDeviceFuncs.h"
#include "AllSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"


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
    int &total_delay = allSynapsesDevice->total_delay[iSyn];

    // Checks if there is an event in the queue.
    return allSynapsesDevice->preSpikeQueue->checkAnEvent(iSyn, total_delay);
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
    return allSynapsesDevice->postSpikeQueue->checkAnEvent(iSyn);
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
 */
__global__ void advanceSpikingSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapsesDeviceProperties* allSynapsesDevice ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= total_synapse_counts )
                return;

        BGSIZE iSyn = synapseIndexMapDevice->incomingSynapseIndexMap[idx];

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

    BGSIZE iSyn = synapseIndexMapDevice->incomingSynapseIndexMap[idx];

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

/* ------------------------------------*\
|* # Device Functions for createSynapse
\* ------------------------------------*/

/*
 * Return 1 if originating neuron is excitatory, -1 otherwise.
 *
 * @param[in] t  synapseType I to I, I to E, E to I, or E to E
 * @return 1 or -1
 */
__device__ int synSign( synapseType t )
{
        switch ( t )
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
    allSynapsesDevice->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

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
    allSynapsesDevice->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

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
    allSynapsesDevice->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

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
    allSynapsesDevice->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

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
 * @param W_d                    Array of synapse weight.
 * @param num_neurons            The total number of neurons.
 */
__device__ void addSpikingSynapse(AllSpikingSynapsesDeviceProperties* allSynapsesDevice, synapseType type, const int neuron_index, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons)
{
    if (allSynapsesDevice->synapse_counts[neuron_index] >= allSynapsesDevice->maxSynapsesPerNeuron) {
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
        createSpikingSynapse(allSynapsesDevice, neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    case classAllDSSynapses:
        createDSSynapse(static_cast<AllDSSynapsesDeviceProperties *>(allSynapsesDevice), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    case classAllSTDPSynapses:
        createSTDPSynapse(static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    case classAllDynamicSTDPSynapses:
        createDynamicSTDPSynapse(static_cast<AllDynamicSTDPSynapsesDeviceProperties *>(allSynapsesDevice), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    default:
        assert(false);
    }
    allSynapsesDevice->W[synapseBegin + synapse_index] = W_d[source_index * num_neurons + dest_index] * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
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
__device__ void eraseSpikingSynapse( AllSpikingSynapsesDeviceProperties* allSynapsesDevice, const int neuron_index, const int synapse_offset, int maxSynapses )
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
__device__ synapseType synType( neuronType* neuron_type_map_d, const int src_neuron, const int dest_neuron )
{
    if ( neuron_type_map_d[src_neuron] == INH && neuron_type_map_d[dest_neuron] == INH )
        return II;
    else if ( neuron_type_map_d[src_neuron] == INH && neuron_type_map_d[dest_neuron] == EXC )
        return IE;
    else if ( neuron_type_map_d[src_neuron] == EXC && neuron_type_map_d[dest_neuron] == INH )
        return EI;
    else if ( neuron_type_map_d[src_neuron] == EXC && neuron_type_map_d[dest_neuron] == EXC )
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
 * @param[in] W_d                Array of synapse weight.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
* @param[in] neuron_type_map_d   Pointer to the neurons type map in device memory.
 * @param[in] totalClusterNeurons  Total number of neurons in the cluster.
 * @param[in] clusterNeuronsBegin  Begin neuron index of the cluster.
 */
__global__ void updateSynapsesWeightsDevice( int num_neurons, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, neuronType* neuron_type_map_d, int totalClusterNeurons, int clusterNeuronsBegin )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_neurons )
        return;

    int adjusted = 0;
    //int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    // Scale and add sign to the areas
    // visit each neuron 'a'
    int iNeuron = idx;
    int dest_neuron = clusterNeuronsBegin + iNeuron;

    // and each destination neuron 'b'
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
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
                    if (W_d[src_neuron * num_neurons + dest_neuron] < 0) {
                        removed++;
                        eraseSpikingSynapse(allSynapsesDevice, iNeuron, synapse_index, maxSynapses);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesDevice->W[iSyn] = W_d[src_neuron * num_neurons
                            + dest_neuron] * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
                    }
                }
                existing_synapses_checked++;
            }
        }

        // if not connected and weight(a,b) > 0, add a new synapse from a to b
        if (!connected && (W_d[src_neuron * num_neurons +  dest_neuron] > 0)) {
            // locate summation point
            BGFLOAT* sum_point = &( allNeuronsDevice->summation_map[iNeuron] );
            added++;

            addSpikingSynapse(allSynapsesDevice, type, iNeuron, src_neuron, dest_neuron, sum_point, deltaT, W_d, num_neurons);
        }
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
__global__ void initSynapsesDevice( int n, AllDSSynapsesDeviceProperties* allSynapsesDevice, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    // create a synapse
    int neuron_index = idx;
    BGFLOAT* sum_point = &( pSummationMap[neuron_index] );
    synapseType type = allSynapsesDevice->type[neuron_index];
    createDSSynapse(allSynapsesDevice, neuron_index, 0, 0, neuron_index, sum_point, deltaT, type );
    allSynapsesDevice->W[neuron_index] = weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
}

/*
 * Creates a EventQueue object in device memory.
 *
 * @param[in] total_synapse_counts  Number of synapses.
 * @param[in/out] pEventQueue       Pointer to the pointer to EventQueue objet
 *                                  where the pointer EventQueue object is stored.
 * @param[in] clusterID             The cluster ID of cluster.
 */
__global__ void allocEventQueueDevice(int total_synapse_counts, EventQueue **pEventQueue, CLUSTER_INDEX_TYPE clusterID)
{
    *pEventQueue = new EventQueue();
    (*pEventQueue)->initEventQueue(total_synapse_counts, clusterID);
}

/*
 * Delete a EventQueue object in device memory.
 *
 * @param[in] pEventQueue          Pointer to the EventQueue object to be deleted.
 */
__global__ void deleteEventQueueDevice(EventQueue *pEventQueue)
{
    if (pEventQueue != NULL) {
        delete pEventQueue;
    }
}

/*
 * Copy event queue data from the buffer to the device between device memories.
 *
 * @param pDstEventQueue       Pointer to the EventQueue object (destination).
 * @param nMaxEvent            The number of event queue (source).
 * @param idxQueue             The index indicating the current time slot in the delayed queue (source).
 * @param pQueueBuffer         Pointer to the collection of event queue (source).
 */
__global__ void copyEventQueueDevice(EventQueue *pDstEventQueue, BGSIZE nMaxEvent, uint32_t idxQueue, BGQUEUE_ELEMENT* pQueueBuffer)
{
    pDstEventQueue->m_nMaxEvent = nMaxEvent;
    pDstEventQueue->m_idxQueue = idxQueue;
    memcpy(pDstEventQueue->m_queueEvent, pQueueBuffer, nMaxEvent * sizeof( BGQUEUE_ELEMENT ));
}

/*
 * Copy event queue data from the buffer to the device between device memories.
 * 
 * @param pSrcEventQueue       Pointer to the EventQueue object (source).
 * @param pQueueBuffer         Pointer to the collection of event queue (destination).
 * @param pDstEventQueue       Pointer to the EventQueue object (destination).
 */
__global__ void copyEventQueueDevice(EventQueue *pSrcEventQueue, BGQUEUE_ELEMENT* pQueueBuffer, EventQueue* pDstEventQueue)
{
    BGSIZE nMaxEvent = pSrcEventQueue->m_nMaxEvent;
    memcpy(pQueueBuffer, pSrcEventQueue->m_queueEvent, nMaxEvent * sizeof( BGQUEUE_ELEMENT ));

    pDstEventQueue->m_nMaxEvent = pSrcEventQueue->m_nMaxEvent;
    pDstEventQueue->m_idxQueue = pSrcEventQueue->m_idxQueue;
}

/*
 * Perform updating preSpikeQueue for one time step.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct
 *                             on device memory.
 */
__global__ void advanceSpikingSynapsesEventQueueDevice(AllSpikingSynapsesDeviceProperties* allSynapsesDevice)
{
    allSynapsesDevice->preSpikeQueue->advanceEventQueue();
}

/*
 * Perform updating postSpikeQueue for one time step.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct
 *                             on device memory.
 */
__global__ void advanceSTDPSynapsesEventQueueDevice(AllSTDPSynapsesDeviceProperties* allSynapsesDevice)
{
    allSynapsesDevice->postSpikeQueue->advanceEventQueue();
}

