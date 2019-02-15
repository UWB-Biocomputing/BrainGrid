#include "AllSpikingSynapses.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#include "AllSynapsesDeviceFuncs.h"
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
void AllSpikingSynapses::resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT)
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
void AllSpikingSynapses::createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);
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

#if defined(USE_GPU)
/*
 *  Advance all the Synapses in the simulation.
 *  Update the state of all synapses for a time step.
 *
 *  @param  allSynapsesProps      Reference to the AllSynapsesProps struct
 *                                 on device memory.
 *  @param  allNeuronsProps       Reference to the allNeurons struct on device memory.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  sim_info               SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllSpikingSynapses::advanceSynapses(void* allSynapsesProps, void* allNeuronsProps, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    BGSIZE total_synapse_counts = m_pSynapsesProps->total_synapse_counts;
    if (total_synapse_counts == 0)
        return;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSpikingSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSpikingSynapsesProps*)allSynapsesProps, iStepOffset );
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 *  @param  allSynapsesProps      Reference to the AllSynapsesProps struct
 *                                 on device memory.
 *  @param  iStep                  Simulation steps to advance.
 */
void AllSpikingSynapses::advanceSpikeQueue(void* allSynapsesProps, int iStep)
{
    advanceSpikingSynapsesEventQueueDevice <<< 1, 1 >>> ((AllSpikingSynapsesProps*)allSynapsesProps, iStep);
}
#else // USE_GPU
/*
 *  Checks if there is an input spike in the queue.
 *
 *  @param  iSyn   Index of the Synapse to connect to.
 *  @param  iStepOffset  Offset from the current simulation step.
 *  @return true if there is an input spike event.
 */
bool AllSpikingSynapses::isSpikeQueue(const BGSIZE iSyn, int iStepOffset)
{
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);
    int &total_delay = pSynapsesProps->total_delay[iSyn];

    // Checks if there is an event in the queue.
    return pSynapsesProps->preSpikeQueue->checkAnEvent(iSyn, total_delay, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit.
 *
 *  @param  iSyn   Index of the Synapse to update.
 *  @param  iStepOffset  Offset from the current simulation step.
 *  @param  iCluster  Cluster ID of cluster where the spike is added.
 */
void AllSpikingSynapses::preSpikeHit(const BGSIZE iSyn, const CLUSTER_INDEX_TYPE iCluster, int iStepOffset)
{
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

    // Add to spike queue
    pSynapsesProps->preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param  iSyn   Index of the Synapse to update.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
void AllSpikingSynapses::postSpikeHit(const BGSIZE iSyn, int iStepOffset)
{
}

/*
 *  Advance all the Synapses in the simulation.
 *
 *  @param  sim_info         SimulationInfo class to read information from.
 *  @param  neurons          The Neuron list to search from.
 *  @param  synapseIndexMap  Pointer to the synapse index map.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSpikingSynapses::advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap, int iStepOffset)
{
    AllSynapses::advanceSynapses(sim_info, neurons, synapseIndexMap, iStepOffset);
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 * @param iStep     simulation steps to advance.
 */
void AllSpikingSynapses::advanceSpikeQueue(int iStep)
{
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

    pSynapsesProps->preSpikeQueue->advanceEventQueue(iStep);
}

/*
 *  Advance one specific Synapse.
 *
 *  @param  iSyn      Index of the Synapse to connect to.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  neurons   The Neuron list to search from.
 *  @param  iStepOffset  Offset from the current simulation step.
 */
void AllSpikingSynapses::advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons * neurons, int iStepOffset)
{
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

    BGFLOAT &decay = pSynapsesProps->decay[iSyn];
    BGFLOAT &psr = pSynapsesProps->psr[iSyn];
    BGFLOAT &summationPoint = *(pSynapsesProps->summationPoint[iSyn]);

    // is an input in the queue?
    if (isSpikeQueue(iSyn, iStepOffset)) {
        changePSR(iSyn, sim_info->deltaT, iStepOffset);
    }

    // decay the post spike response
    psr *= decay;
    // and apply it to the summation point
    summationPoint += psr;
}

/*
 *  Calculate the post synapse response after a spike.
 *
 *  @param  iSyn        Index of the synapse to set.
 *  @param  deltaT      Inner simulation step duration.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSpikingSynapses::changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, int iStepOffset)
{
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

    BGFLOAT &psr = pSynapsesProps->psr[iSyn];
    BGFLOAT &W = pSynapsesProps->W[iSyn];
    BGFLOAT &decay = pSynapsesProps->decay[iSyn];

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
        AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(m_pSynapsesProps);

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
bool AllSpikingSynapses::allowBackPropagation()
{
    return false;
}

#if defined(USE_GPU)
/*
 *  Set some parameters used for advanceSynapsesDevice.
 */
void AllSpikingSynapses::setAdvanceSynapsesDeviceParams()
{
    setSynapseClassID();
}

/**
 *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
 *  The class ID will be set to classSynapses_d in device memory,
 *  and the classSynapses_d will be referred to call a device function for the
 *  particular synapse class.
 *  Because we cannot use virtual function (Polymorphism) in device functions,
 *  we use this scheme.
 *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
 *  (see issue#137).
 */
void AllSpikingSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllSpikingSynapses;

    checkCudaErrors( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}
#endif // USE_GPU
