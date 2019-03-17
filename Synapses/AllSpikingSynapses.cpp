#include "AllSpikingSynapses.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif // USE_GPU

// Default constructor
AllSpikingSynapses::AllSpikingSynapses() 
{
}

AllSpikingSynapses::~AllSpikingSynapses()
{
}

/*
 *  Create and setup synapses properties.
 */
void AllSpikingSynapses::createSynapsesProps()
{
    m_pSynapsesProps = new AllSpikingSynapsesProps();
}

/*
 *  Sets the data for Synapses to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingSynapses::deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info)
{
    AllSynapses::deserialize(input, neurons, clr_info);

    dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps)->preSpikeQueue->deserialize(input);
}

/*
 *  Write the synapses data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingSynapses::serialize(ostream& output, const ClusterInfo *clr_info)
{
    AllSynapses::serialize(output, clr_info);

    dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps)->preSpikeQueue->serialize(output);
}

/*
 *  Reset time varying state vars and recompute decay.
 *
 *  @param  iSyn     Index of the synapse to set.
 *  @param  deltaT   Inner simulation step duration
 */
CUDA_CALLABLE void AllSpikingSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
{
    AllSynapses::resetSynapse(iSyn, deltaT);

    assert( updateDecay(iSyn, deltaT) );
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
CUDA_CALLABLE void AllSpikingSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    AllSpikingSynapsesProps *pSynapsesProps = reinterpret_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);
    BGFLOAT delay;

    pSynapsesProps->in_use[iSyn] = true;
    pSynapsesProps->summationPoint[iSyn] = sum_point;
    pSynapsesProps->destNeuronLayoutIndex[iSyn] = dest_index;
    pSynapsesProps->sourceNeuronLayoutIndex[iSyn] = source_index;
    pSynapsesProps->W[iSyn] = synSign(type) * 10.0e-9;
    pSynapsesProps->type[iSyn] = type;
    pSynapsesProps->tau[iSyn] = DEFAULT_tau;

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

    pSynapsesProps->tau[iSyn] = tau;
    pSynapsesProps->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    assert( pSynapsesProps->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY );

    // initializes the queues for the Synapses
    pSynapsesProps->preSpikeQueue->clearAnEvent(iSyn);

    // reset time varying state vars and recompute decay
    resetSynapse(iSyn, deltaT);
}

/*
 *  Checks if there is an input spike in the queue.
 *
 *  @param  iSyn             Index of the Synapse to connect to.
 *  @param  iStepOffset      Offset from the current global simulation step.
 *  @param  pSynapsesProps   Pointer to the synapses properties.
 *  @return true if there is an input spike event.
 */
CUDA_CALLABLE bool AllSpikingSynapses::isSpikeQueue(const BGSIZE iSyn, int iStepOffset, AllSpikingSynapsesProps* pSynapsesProps)
{
    int &total_delay = pSynapsesProps->total_delay[iSyn];

    // Checks if there is an event in the queue.
    return pSynapsesProps->preSpikeQueue->checkAnEvent(iSyn, total_delay, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit.
 *
 *  @param  iSyn             Index of the Synapse to update.
 *  @param  iStepOffset      Offset from the current simulation step.
 *  @param  iCluster         Cluster ID of cluster where the spike is added.
 */
CUDA_CALLABLE void AllSpikingSynapses::preSpikeHit(const BGSIZE iSyn, const CLUSTER_INDEX_TYPE iCluster, int iStepOffset)
{
    AllSpikingSynapsesProps *pSynapsesProps = reinterpret_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

    // Add to spike queue
    pSynapsesProps->preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to update.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
CUDA_CALLABLE void AllSpikingSynapses::postSpikeHit(const BGSIZE iSyn, int iStepOffset)
{
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 * @param iStep            simulation steps to advance.
 */
CUDA_CALLABLE void AllSpikingSynapses::advanceSpikeQueue(int iStep)
{
    AllSpikingSynapsesProps *pSynapsesProps = reinterpret_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

    pSynapsesProps->preSpikeQueue->advanceEventQueue(iStep);
}

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
CUDA_CALLABLE void AllSpikingSynapses::advanceSynapse(const BGSIZE iSyn, const BGFLOAT deltaT, IAllNeurons * neurons, uint64_t simulationStep, int iStepOffset, int maxSpikes, IAllNeuronsProps* pINeuronsProps)
{
    AllSpikingSynapsesProps *pSynapsesProps = reinterpret_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

    BGFLOAT &decay = pSynapsesProps->decay[iSyn];
    BGFLOAT &psr = pSynapsesProps->psr[iSyn];

    // is an input in the queue?
    if (isSpikeQueue(iSyn, iStepOffset, pSynapsesProps)) {
        changePSR(iSyn, deltaT, simulationStep, pSynapsesProps);
    }

    // decay the post spike response
    psr *= decay;
}

/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn             Index of the synapse to set.
 *  @param  deltaT           Inner simulation step duration.
 *  @param  simulationStep   The current simulation step.
 *  @param  pSynapsesProps   Pointer to the synapses properties.
 */
CUDA_CALLABLE void AllSpikingSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, uint64_t simulationStep, AllSpikingSynapsesProps* pSynapsesProps)
{
    BGFLOAT &psr = pSynapsesProps->psr[iSyn];
    BGFLOAT &W = pSynapsesProps->W[iSyn];
    BGFLOAT &decay = pSynapsesProps->decay[iSyn];

    psr += ( W / decay );    // calculate psr
}

/*
 *  Updates the decay if the synapse selected.
 *
 *  @param  iSyn    Index of the synapse to set.
 *  @param  deltaT  Inner simulation step duration
 */
CUDA_CALLABLE bool AllSpikingSynapses::updateDecay(const BGSIZE iSyn, const BGFLOAT deltaT)
{
        AllSpikingSynapsesProps *pSynapsesProps = reinterpret_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

        BGFLOAT &tau = pSynapsesProps->tau[iSyn];
        BGFLOAT &decay = pSynapsesProps->decay[iSyn];

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
CUDA_CALLABLE bool AllSpikingSynapses::allowBackPropagation()
{
    return false;
}

#if defined(USE_GPU)

/*
 *  Create a AllSynapses class object in device
 *
 *  @param pAllSynapses_d       Device memory address to save the pointer of created AllSynapses object.
 *  @param pAllSynapsesProps_d  Pointer to the synapses properties in device memory.
 */
void AllSpikingSynapses::createAllSynapsesInDevice(IAllSynapses** pAllSynapses_d, IAllSynapsesProps *pAllSynapsesProps_d)
{
    IAllSynapses **pAllSynapses_t; // temporary buffer to save pointer to IAllSynapses object.

    // allocate device memory for the buffer.
    checkCudaErrors( cudaMalloc( ( void ** ) &pAllSynapses_t, sizeof( IAllSynapses * ) ) );

    // create an AllSynapses object in device memory.
    allocAllSpikingSynapsesDevice <<< 1, 1 >>> ( pAllSynapses_t, pAllSynapsesProps_d );

    // save the pointer of the object.
    checkCudaErrors( cudaMemcpy ( pAllSynapses_d, pAllSynapses_t, sizeof( IAllSynapses * ), cudaMemcpyDeviceToHost ) );

    // free device memory for the buffer.
    checkCudaErrors( cudaFree( pAllSynapses_t ) );
}

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllSpikingSynapsesDevice(IAllSynapses **pAllSynapses, IAllSynapsesProps *pAllSynapsesProps)
{
    *pAllSynapses = new AllSpikingSynapses();
    (*pAllSynapses)->setSynapsesProps(pAllSynapsesProps);
}

/*
 * Perform updating preSpikeQueue for one time step.
 *
 *  @param  iStep              Simulation steps to advance.
 *  @param  synapsesDevice     Pointer to the synapses object in device memory.
 */
__global__ void advanceSpikeQueueDevice(int iStep, IAllSynapses* synapsesDevice)
{
    reinterpret_cast<AllSpikingSynapses*>(synapsesDevice)->advanceSpikeQueue(iStep);
}

#endif // USE_GPU

