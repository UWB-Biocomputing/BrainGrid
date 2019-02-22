#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"

// Default constructor
CUDA_CALLABLE AllSpikingNeurons::AllSpikingNeurons()
{
}

CUDA_CALLABLE AllSpikingNeurons::~AllSpikingNeurons()
{
}

#if !defined(USE_GPU)
/*
 *  Update internal state of the indexed Neuron (called by every simulation step).
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses         The Synapse list to search from.
 *  @param  sim_info         SimulationInfo class to read information from.
 *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
 *  @param  clr_info         ClusterInfo class to read information from.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSpikingNeurons::advanceNeurons(IAllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap, const ClusterInfo *clr_info, int iStepOffset)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(spSynapses.m_pSynapsesProps);
    bool *hasFired = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->hasFired;
    int *spikeCount = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCount; 

    // For each neuron in the network
    for (int idx = clr_info->totalClusterNeurons - 1; idx >= 0; --idx) {
        // advance neurons
        advanceNeuron(idx, sim_info, clr_info, iStepOffset);

        // notify outgoing/incomming synapses if neuron has fired
        if (hasFired[idx]) {
            DEBUG_MID(cout << " !! Neuron" << idx << "has Fired @ t: " << (g_simulationStep + iStepOffset) * sim_info->deltaT << endl;)

            assert( spikeCount[idx] < max_spikes );

            if (pSynapsesProps->total_synapse_counts != 0) {
                // notify outgoing synapses
                BGSIZE synapse_counts;

                synapse_counts = synapseIndexMap->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    BGSIZE beginIndex = synapseIndexMap->outgoingSynapseBegin[idx];
                    OUTGOING_SYNAPSE_INDEX_TYPE* outgoingMap_begin = &( synapseIndexMap->outgoingSynapseIndexMap[beginIndex] );
                    for ( BGSIZE i = 0; i < synapse_counts; i++ ) {
                        OUTGOING_SYNAPSE_INDEX_TYPE idx = outgoingMap_begin[i];
                        // outgoing synapse index consists of cluster index + synapse index
                        CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
                        BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
                        spSynapses.preSpikeHit(iSyn, iCluster, iStepOffset);
                    }
                }

                // notify incomming synapses
                synapse_counts = synapseIndexMap->incomingSynapseIndexMap[idx];

                if (spSynapses.allowBackPropagation() && synapse_counts != 0) {
                    int beginIndex = synapseIndexMap->incomingSynapseBegin[idx];
                    BGSIZE* incomingMap_begin = &( synapseIndexMap->incomingSynapseIndexMap[beginIndex] );
          
                    for ( BGSIZE i = 0; i < synapse_counts; i++ ) {
                        BGSIZE iSyn = incomingMap_begin[i];
                        spSynapses.postSpikeHit(iSyn, iStepOffset);
                    }
                }
            }

            hasFired[idx] = false;
        }
    }
}

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  iStepOffset      Offset from the current simulation step.
 */
void AllSpikingNeurons::fire(const int index, const SimulationInfo *sim_info, int iStepOffset) const
{
    bool *hasFired = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->hasFired;
    int *spikeCountOffset = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCountOffset;
    int *spikeCount = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCount; 
    uint64_t **spike_history = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spike_history;

    // Note that the neuron has fired!
    hasFired[index] = true;
    
    // record spike time
    uint64_t simulationStep = g_simulationStep + iStepOffset;
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    int idxSp = (spikeCount[index] + spikeCountOffset[index]) % max_spikes;
    spike_history[index][idxSp] = simulationStep;

    DEBUG_SYNAPSE(
        cout << "AllSpikingNeurons::fire:" << endl;
        cout << "          index: " << index << endl;
        cout << "          simulationStep: " << simulationStep << endl << endl;
    );
    
    // increment spike count and total spike count
    spikeCount[index]++;
}

/*
 *  Get the spike history of neuron[index] at the location offIndex.
 *
 *  @param  index            Index of the neuron to get spike history.
 *  @param  offIndex         Offset of the history buffer to get from.
 *  @param  sim_info         SimulationInfo class to read information from.
 */
uint64_t AllSpikingNeurons::getSpikeHistory(int index, int offIndex, const SimulationInfo *sim_info)
{
    int *spikeCountOffset = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCountOffset;
    int *spikeCount = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCount; 
    uint64_t **spike_history = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spike_history;

    // offIndex is a minus offset
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    int idxSp = (spikeCount[index] + spikeCountOffset[index] +  max_spikes + offIndex) % max_spikes;
    return spike_history[index][idxSp];
}
#endif // !USE_GPU

#if defined(USE_GPU)
/*
 *  Set some parameters used for advanceNeuronsDevice.
 *  Currently we set the two member variables: m_fpPreSpikeHit_h and m_fpPostSpikeHit_h.
 *  These are function pointers for PreSpikeHit and PostSpikeHit device functions
 *  respectively, and these functions are called from advanceNeuronsDevice device
 *  function. We use this scheme because we cannot not use virtual function (Polymorphism)
 *  in device functions.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 */
void AllSpikingNeurons::setAdvanceNeuronsDeviceParams(IAllSynapses &synapses)
{
    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    m_fAllowBackPropagation = spSynapses.allowBackPropagation();
}
#endif // USE_GPU
