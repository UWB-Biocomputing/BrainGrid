#include "AllSTDPSynapses.h"
#include "IAllNeurons.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif // USE_GPU

// Default constructor
CUDA_CALLABLE AllSTDPSynapses::AllSTDPSynapses()
{
}

CUDA_CALLABLE AllSTDPSynapses::~AllSTDPSynapses()
{
}

/*
 *  Create and setup synapses properties.
 */
void AllSTDPSynapses::createSynapsesProps()
{
    m_pSynapsesProps = new AllSTDPSynapsesProps();
}

/*
 *  Sets the data for Synapses to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSTDPSynapses::deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info)
{
    AllSTDPSynapsesProps *pSynapsesProps = dynamic_cast<AllSTDPSynapsesProps*>(m_pSynapsesProps);

    AllSpikingSynapses::deserialize(input, neurons, clr_info);

    pSynapsesProps->postSpikeQueue->deserialize(input);
}

/*
 *  Write the synapses data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSTDPSynapses::serialize(ostream& output, const ClusterInfo *clr_info)
{
    AllSTDPSynapsesProps *pSynapsesProps = dynamic_cast<AllSTDPSynapsesProps*>(m_pSynapsesProps);

    AllSpikingSynapses::serialize(output, clr_info);

    pSynapsesProps->postSpikeQueue->serialize(output);
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn            Index of the synapse to set.
 *  @param  deltaT          Inner simulation step duration
 */
CUDA_CALLABLE void AllSTDPSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    AllSpikingSynapses::resetSynapse(iSyn, deltaT);
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
CUDA_CALLABLE void AllSTDPSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    AllSTDPSynapsesProps *pSynapsesProps = reinterpret_cast<AllSTDPSynapsesProps*>(m_pSynapsesProps);

    AllSpikingSynapses::createSynapse(iSyn, source_index, dest_index, sum_point, deltaT, type);

    pSynapsesProps->Apos[iSyn] = 0.5;
    pSynapsesProps->Aneg[iSyn] = -0.5;
    pSynapsesProps->STDPgap[iSyn] = 2e-3;

    pSynapsesProps->total_delayPost[iSyn] = 0;

    pSynapsesProps->tauspost[iSyn] = 75e-3;
    pSynapsesProps->tauspre[iSyn] = 34e-3;

    pSynapsesProps->taupos[iSyn] = 15e-3;
    pSynapsesProps->tauneg[iSyn] = 35e-3;
    pSynapsesProps->Wex[iSyn] = 1.0;

    pSynapsesProps->mupos[iSyn] = 0;
    pSynapsesProps->muneg[iSyn] = 0;

    pSynapsesProps->useFroemkeDanSTDP[iSyn] = true;

    // initializes the queues for the Synapses
    pSynapsesProps->postSpikeQueue->clearAnEvent(iSyn);
}

#if defined(USE_GPU)

/*
 *  Create a AllSynapses class object in device
 *
 *  @param pAllSynapses_d       Device memory address to save the pointer of created AllSynapses object.
 *  @param pAllSynapsesProps_d  Pointer to the synapses properties in device memory.
 */
void AllSTDPSynapses::createAllSynapsesInDevice(IAllSynapses** pAllSynapses_d, IAllSynapsesProps *pAllSynapsesProps_d)
{
    IAllSynapses **pAllSynapses_t; // temporary buffer to save pointer to IAllSynapses object.

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pAllSynapses_t, sizeof( IAllSynapses * ) ) );

    // create an AllSynapses object in device memory.
    allocAllSTDPSynapsesDevice <<< 1, 1 >>> ( pAllSynapses_t, pAllSynapsesProps_d );

    // save the pointer of the object.
    checkCudaErrors( cudaMemcpy ( pAllSynapses_d, pAllSynapses_t, sizeof( IAllSynapses * ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pAllSynapses_t ) );
}

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllSTDPSynapsesDevice(IAllSynapses **pAllSynapses, IAllSynapsesProps *pAllSynapsesProps)
{
    *pAllSynapses = new AllSTDPSynapses();
    (*pAllSynapses)->setSynapsesProps(pAllSynapsesProps);
}

#endif // USE_GPU

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn             Index of the Synapse to connect to.
 *  @param  deltaT           Inner simulation step duration.
 *  @param  neurons          The Neuron list to search from.
 *  @param  simulationStep   The current simulation step.
 *  @param  iStepOffset      Offset from the current global simulation step.
 *  @param  maxSpikes        Maximum number of spikes per neuron per epoch.
 *  @param  pINeuronsProps   Pointer to the neurons properties.
 */
CUDA_CALLABLE void AllSTDPSynapses::advanceSynapse(const BGSIZE iSyn, const BGFLOAT deltaT, IAllNeurons *neurons, uint64_t simulationStep, int iStepOffset, int maxSpikes, IAllNeuronsProps* pINeuronsProps)
{
    AllSTDPSynapsesProps *pSynapsesProps = reinterpret_cast<AllSTDPSynapsesProps*>(m_pSynapsesProps);

    BGFLOAT &decay = pSynapsesProps->decay[iSyn];
    BGFLOAT &psr = pSynapsesProps->psr[iSyn];

    // is an input in the queue?
    bool fPre = isSpikeQueue(iSyn, iStepOffset, pSynapsesProps); 
    bool fPost = isSpikeQueuePost(iSyn, iStepOffset, pSynapsesProps);

    if (fPre || fPost) {
        BGFLOAT &tauspre = pSynapsesProps->tauspre[iSyn];
        BGFLOAT &tauspost = pSynapsesProps->tauspost[iSyn];
        BGFLOAT &taupos = pSynapsesProps->taupos[iSyn];
        BGFLOAT &tauneg = pSynapsesProps->tauneg[iSyn];
        int &total_delay = pSynapsesProps->total_delay[iSyn];
        bool &useFroemkeDanSTDP = pSynapsesProps->useFroemkeDanSTDP[iSyn];

        AllSpikingNeurons* spNeurons = reinterpret_cast<AllSpikingNeurons*>(neurons);

        // pre and post neurons index
        int idxPre = pSynapsesProps->sourceNeuronLayoutIndex[iSyn];
        int idxPost = pSynapsesProps->destNeuronLayoutIndex[iSyn];
        uint64_t spikeHistory, spikeHistory2;
        BGFLOAT delta;
        BGFLOAT epre, epost;

        if (fPre) {	// preSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time 
            // just one before the last spike.
            spikeHistory = spNeurons->getSpikeHistory(idxPre, -2, maxSpikes, pINeuronsProps);
            if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = ((int64_t)simulationStep - spikeHistory) * deltaT;
                epre = 1.0 - exp(-delta / tauspre);
            } else {
                epre = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // pre-post spikes
            int offIndex = -1;	// last spike
            while (true) {
                spikeHistory = spNeurons->getSpikeHistory(idxPost, offIndex, maxSpikes, pINeuronsProps);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between pre-post spikes
                // (include pre-synaptic transmission delay)
                delta = (spikeHistory - (int64_t)simulationStep) * deltaT;

                DEBUG_SYNAPSE(
                    printf("AllSTDPSynapses::advanceSynapse: fPre\n");
                    printf("          iSyn: %d\n", iSyn);
                    printf("          idxPre: %d\n", idxPre);
                    printf("          idxPost: %d\n", idxPost);
                    printf("          spikeHistory: %ld\n", spikeHistory);
                    printf("          simulationStep: %ld\n", simulationStep);
                    printf("          delta: %f\n\n", delta);
                );

                if (delta <= -3.0 * tauneg)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = spNeurons->getSpikeHistory(idxPost, offIndex-1, maxSpikes, pINeuronsProps);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epost = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspost);
                } else {
                    epost = 1.0;
                }
                stdpLearning(iSyn, delta, epost, epre, pSynapsesProps);
                --offIndex;
            }

            changePSR(iSyn, deltaT, simulationStep, pSynapsesProps);
        }

        if (fPost) {	// postSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time
            // just one before the last spike.
            spikeHistory = spNeurons->getSpikeHistory(idxPost, -2, maxSpikes, pINeuronsProps);
            if (spikeHistory != ULONG_MAX && useFroemkeDanSTDP) {
                // delta will include the transmission delay
                delta = ((int64_t)simulationStep - spikeHistory) * deltaT;
                epost = 1.0 - exp(-delta / tauspost);
            } else {
                epost = 1.0;
            }

            // call the learning function stdpLearning() for each pair of
            // post-pre spikes
            int offIndex = -1;	// last spike
            while (true) {
                spikeHistory = spNeurons->getSpikeHistory(idxPre, offIndex, maxSpikes, pINeuronsProps);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between post-pre spikes
                delta = ((int64_t)simulationStep - spikeHistory - total_delay) * deltaT;

                DEBUG_SYNAPSE(
                    printf("AllSTDPSynapses::advanceSynapse: fPost\n");
                    printf("          iSyn: %d\n", iSyn);
                    printf("          idxPre: %d\n", idxPre);
                    printf("          idxPost: %d\n", idxPost);
                    printf("          spikeHistory: %ld\n", spikeHistory);
                    printf("          simulationStep: %ld\n", simulationStep);
                    printf("          delta: %f\n\n", delta);
                );

                if (delta <= 0 || delta >= 3.0 * taupos)
                    break;
                if (useFroemkeDanSTDP) {
                    spikeHistory2 = spNeurons->getSpikeHistory(idxPre, offIndex-1, maxSpikes, pINeuronsProps);
                    if (spikeHistory2 == ULONG_MAX)
                        break;                
                    epre = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspre);
                } else {
                    epre = 1.0;
                }
                stdpLearning(iSyn, delta, epost, epre, pSynapsesProps);
                --offIndex;
            }
        }
    }

    // decay the post spike response
    psr *= decay;
}

/*
 *  Adjust synapse weight according to the Spike-timing-dependent synaptic modification 
 *  induced by natural spike trains
 *
 *  @param  iSyn             Index of the synapse to set.
 *  @param  delta            Pre/post synaptic spike interval.
 *  @param  epost            Params for the rule given in Froemke and Dan (2002).
 *  @param  epre             Params for the rule given in Froemke and Dan (2002).
 *  @param  pISynapsesProps  Pointer to the synapses properties.
 */
CUDA_CALLABLE void AllSTDPSynapses::stdpLearning(const BGSIZE iSyn, double delta, double epost, double epre, AllSTDPSynapsesProps* pSynapsesProps)
{
    BGFLOAT STDPgap = pSynapsesProps->STDPgap[iSyn];
    BGFLOAT muneg = pSynapsesProps->muneg[iSyn];
    BGFLOAT mupos = pSynapsesProps->mupos[iSyn];
    BGFLOAT tauneg = pSynapsesProps->tauneg[iSyn];
    BGFLOAT taupos = pSynapsesProps->taupos[iSyn];
    BGFLOAT Aneg = pSynapsesProps->Aneg[iSyn];
    BGFLOAT Apos = pSynapsesProps->Apos[iSyn];
    BGFLOAT Wex = pSynapsesProps->Wex[iSyn];
    BGFLOAT &W = pSynapsesProps->W[iSyn];
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
        printf("AllSTDPSynapses::stdpLearning: fPost\n");
        printf("          iSyn: %d\n", iSyn);
        printf("          delta: %f\n", delta);
        printf("          epre: %f\n", epre);
        printf("          epost: %f\n", epost);
        printf("          dw: %f\n", dw);
        printf("          W: %f\n\n", W);
    );
}

/*
 *  Checks if there is an input spike in the queue (for back propagation).
 *
 *  @param  iSyn             Index of the Synapse to connect to.
 *  @param  iStepOffset      Offset from the current simulation step.
 *  @param  pSynapsesProps   Pointer to the synapses properties.
 *  @return true if there is an input spike event.
 */
CUDA_CALLABLE bool AllSTDPSynapses::isSpikeQueuePost(const BGSIZE iSyn, int iStepOffset, AllSpikingSynapsesProps* pSpikingSynapsesProps)
{
    AllSTDPSynapsesProps *pSynapsesProps = reinterpret_cast<AllSTDPSynapsesProps*>(pSpikingSynapsesProps);

    // Checks if there is an event in the queue
    return pSynapsesProps->postSpikeQueue->checkAnEvent(iSyn, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
CUDA_CALLABLE void AllSTDPSynapses::postSpikeHit(const BGSIZE iSyn, int iStepOffset)
{
    AllSTDPSynapsesProps *pSynapsesProps = reinterpret_cast<AllSTDPSynapsesProps*>(m_pSynapsesProps);

    int &total_delay = pSynapsesProps->total_delayPost[iSyn];

    // Add to spike queue
    pSynapsesProps->postSpikeQueue->addAnEvent(iSyn, total_delay, iStepOffset);
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 * @param iStep            simulation steps to advance.
 */
CUDA_CALLABLE void AllSTDPSynapses::advanceSpikeQueue(int iStep)
{
    AllSTDPSynapsesProps *pSynapsesProps = reinterpret_cast<AllSTDPSynapsesProps*>(m_pSynapsesProps);

    AllSpikingSynapses::advanceSpikeQueue(iStep);

    pSynapsesProps->postSpikeQueue->advanceEventQueue(iStep);
}

/*
 *  Check if the back propagation (notify a spike event to the pre neuron)
 *  is allowed in the synapse class.
 *
 *  @retrun true if the back propagation is allowed.
 */
CUDA_CALLABLE bool AllSTDPSynapses::allowBackPropagation()
{
    return true;
}


