/*
 * AllSpikingNeurons_d.cu
 *
 */

#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "GPUSpikingCluster.h"
#include <helper_cuda.h>

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsProperties   Reference to the AllSpikingNeuronsProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingNeurons::copyDeviceSpikeHistoryToHost( AllSpikingNeuronsProperties& allNeuronsProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
        int numNeurons = clr_info->totalClusterNeurons;
        uint64_t* pSpikeHistory[numNeurons];
        checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProperties.spike_history, numNeurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );

        int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
        for (int i = 0; i < numNeurons; i++) {
                checkCudaErrors( cudaMemcpy ( spike_history[i], pSpikeHistory[i],
                        max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsProperties   Reference to the AllSpikingNeuronsProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingNeurons::copyDeviceSpikeCountsToHost( AllSpikingNeuronsProperties& allNeuronsProperties, const ClusterInfo *clr_info ) 
{
        int numNeurons = clr_info->totalClusterNeurons;

        checkCudaErrors( cudaMemcpy ( spikeCount, allNeuronsProperties.spikeCount, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( spikeCountOffset, allNeuronsProperties.spikeCountOffset, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Clear the spike counts out of all Neurons.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 *  @param  clr       Cluster class to read information from.
 */
void AllSpikingNeurons::clearSpikeCounts(const SimulationInfo *sim_info, const ClusterInfo *clr_info, Cluster *clr)
{
    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    // clear spike counts in host memory
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    int numNeurons = clr_info->totalClusterNeurons;

    for (int i = 0; i < numNeurons; i++) {
        spikeCountOffset[i] = (spikeCount[i] + spikeCountOffset[i]) % max_spikes;
        spikeCount[i] = 0;
    }

    // clear spike counts in device memory
    AllSpikingNeuronsProperties allNeuronsProperties;
    AllSpikingNeuronsProperties *allNeuronsDeviceProperties = dynamic_cast<GPUSpikingCluster*>(clr)->m_allNeuronsProperties;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllSpikingNeuronsProperties ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemset( allNeuronsProperties.spikeCount, 0, numNeurons * sizeof( int ) ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProperties.spikeCountOffset, spikeCountOffset, numNeurons * sizeof( int ), cudaMemcpyHostToDevice ) );
}

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
