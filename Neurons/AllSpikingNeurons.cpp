#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#if defined(USE_GPU)
#include "AllSynapsesDeviceFuncs.h"
#include <helper_cuda.h>
#endif

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
    int maxSpikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(spSynapses.m_pSynapsesProps);
    bool *hasFired = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->hasFired;
    int *spikeCount = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCount; 
    const BGFLOAT deltaT = sim_info->deltaT;
    Norm *normRand = clr_info->normRand;
    uint64_t simulationStep = g_simulationStep + iStepOffset;

    // For each neuron in the network
    for (int idx = clr_info->totalClusterNeurons - 1; idx >= 0; --idx) {
        // advance neurons
        advanceNeuron(idx, maxSpikes, deltaT, simulationStep, m_pNeuronsProps, normRand);

        // notify outgoing/incomming synapses if neuron has fired
        if (hasFired[idx]) {
            DEBUG_MID(cout << " !! Neuron" << idx << "has Fired @ t: " << (simulationStep) * sim_info->deltaT << endl;)

            assert( spikeCount[idx] < maxSpikes );

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
 *  Get the spike history of neuron[index] at the location offIndex.
 *
 *  @param  index            Index of the neuron to get spike history.
 *  @param  offIndex         Offset of the history buffer to get from.
 *  @param  maxSpikes        Maximum number of spikes per neuron per epoch.
 */
CUDA_CALLABLE uint64_t AllSpikingNeurons::getSpikeHistory(int index, int offIndex, int maxSpikes)
{
    int *spikeCountOffset = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCountOffset;
    int *spikeCount = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCount; 
    uint64_t **spike_history = dynamic_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spike_history;

    // offIndex is a minus offset
    int idxSp = (spikeCount[index] + spikeCountOffset[index] +  maxSpikes + offIndex) % maxSpikes;

    return spike_history[index][idxSp];
}

#else // USE_GPU

/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsProps       Reference to the allNeuronsProps struct
 *                                 on device memory.
 *  @param  allSynapsesProps      Reference to the allSynapsesProps struct
 *                                 on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 *  @param  iStepOffset            Offset from the current simulation step.
 *  @param  neuronsDevice          Pointer to the Neurons object in device memory.
 */
void AllSpikingNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsProps, void* allSynapsesProps, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset, IAllNeurons* neuronsDevice )
{
    DEBUG (
    int deviceId;
    checkCudaErrors( cudaGetDevice( &deviceId ) );
    assert(deviceId == clr_info->deviceId);
    ); // end DEBUG

    int neuron_count = clr_info->totalClusterNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (IAllNeuronsProps *)allNeuronsProps, (AllSpikingSynapsesProps*)allSynapsesProps, synapseIndexMapDevice, m_fAllowBackPropagation, iStepOffset, neuronsDevice );
}

/* -------------------------------------*\
|* # Device Functions for advanceNeurons
\* -------------------------------------*/

/*
 *  Prepares Synapse for a spike hit.
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesProps     Pointer to AllSpikingSynapsesProps structures
 *                                   on device memory.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__device__ void preSpikingSynapsesSpikeHitDevice( const BGSIZE iSyn, AllSpikingSynapsesProps* allSynapsesProps, const CLUSTER_INDEX_TYPE iCluster, int iStepOffset ) {
        allSynapsesProps->preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesProps     Pointer to AllSpikingSynapsesProps structures
 *                                   on device memory.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__device__ void postSpikingSynapsesSpikeHitDevice( const BGSIZE iSyn, AllSpikingSynapsesProps* allSynapsesProps, int iStepOffset ) {
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesProps     Pointer to AllSTDPSynapsesProps structures
 *                                   on device memory.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__device__ void postSTDPSynapseSpikeHitDevice( const BGSIZE iSyn, AllSTDPSynapsesProps* allSynapsesProps, int iStepOffset ) {
        int total_delay = allSynapsesProps->total_delayPost[iSyn];
        allSynapsesProps->postSpikeQueue->addAnEvent(iSyn, total_delay, iStepOffset);
}

/* -------------------------------------*\
|* # Global Functions for advanceNeurons
\* -------------------------------------*/

/*
 *  CUDA code for advancing LIF neurons
 *
 *  @param[in] totalNeurons          Number of neurons.
 *  @param[in] maxSynapses           Maximum number of synapses per neuron.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] randNoise             Pointer to device random noise array.
 *  @param[in] pINeuronsProps        Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesProps      Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 *  @param[in] neuronsDevice         Pointer to the Neurons object in device memory.
 */
__global__ void advanceNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, IAllNeuronsProps* pINeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, SynapseIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation, int iStepOffset, IAllNeurons* neuronsDevice ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalNeurons )
                return;

        AllSpikingNeuronsProps *pNeuronsProps = reinterpret_cast<AllSpikingNeuronsProps*>(pINeuronsProps);
        bool &hasFired = pNeuronsProps->hasFired[idx];

        static_cast<AllSpikingNeurons*>(neuronsDevice)->advanceNeuron( idx, maxSpikes, deltaT, simulationStep + iStepOffset, pINeuronsProps, randNoise );

        // notify outgoing synapses of spike
        if (hasFired) {
                BGSIZE synapse_counts = synapseIndexMapDevice->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->outgoingSynapseBegin[idx];
                    // get the memory location of where that list begins
                    OUTGOING_SYNAPSE_INDEX_TYPE* outgoingMap_begin = &(synapseIndexMapDevice->outgoingSynapseIndexMap[beginIndex]);

                    // for each synapse, let them know we have fired
                    for (BGSIZE i = 0; i < synapse_counts; i++) {
                        OUTGOING_SYNAPSE_INDEX_TYPE idx = outgoingMap_begin[i];
                        // outgoing synapse index consists of cluster index + synapse index
                        CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
                        BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
                        preSpikingSynapsesSpikeHitDevice(iSyn, allSynapsesProps, iCluster, iStepOffset);
                    }
                }

                // notify incomming synapses of spike
                synapse_counts = synapseIndexMapDevice->incomingSynapseCount[idx];
                if (fAllowBackPropagation && synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->incomingSynapseBegin[idx];
                    // get the memory location of where that list begins
                    BGSIZE* incomingMap_begin = &(synapseIndexMapDevice->incomingSynapseIndexMap[beginIndex]);

                    // for each synapse, let them know we have fired
                    switch (classSynapses_d) {
                    case classAllSTDPSynapses:
                    case classAllDynamicSTDPSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSTDPSynapseSpikeHitDevice(incomingMap_begin[i], static_cast<AllSTDPSynapsesProps *>(allSynapsesProps), iStepOffset);
                        } // end for
                        break;

                    case classAllSpikingSynapses:
                    case classAllDSSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSpikingSynapsesSpikeHitDevice(incomingMap_begin[i], allSynapsesProps, iStepOffset);
                        } // end for
                        break;

                    default:
                        assert(false);
                    } // end switch
                }
        }
}

#endif // !USE_GPU

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index                 Index of the Neuron to update.
 *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param  deltaT                Inner simulation step duration.
 *  @param  simulationStep        The current simulation step.
 *  @param  pINeuronsProps        Pointer to the neurons properties.
 */
CUDA_CALLABLE void AllSpikingNeurons::fire(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps) const
{
    bool *hasFired = reinterpret_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->hasFired;
    int *spikeCountOffset = reinterpret_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCountOffset;
    int *spikeCount = reinterpret_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spikeCount; 
    uint64_t **spike_history = reinterpret_cast<AllSpikingNeuronsProps*>(m_pNeuronsProps)->spike_history;

    // Note that the neuron has fired!
    hasFired[index] = true;
    
    // record spike time
    int idxSp = (spikeCount[index] + spikeCountOffset[index]) % maxSpikes;
    spike_history[index][idxSp] = simulationStep;

    DEBUG_SYNAPSE(
        printf("AllSpikingNeurons::fire:\n");
        printf("          index: %d\n", index);
        printf("          simulationStep: %ld\n\n", simulationStep);
    );
    
    // increment spike count and total spike count
    spikeCount[index]++;
}

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
