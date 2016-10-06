/*
 * AllSpikingNeurons_d.cu
 *
 */

#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"
#include "Book.h"

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllSpikingNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingNeurons::copyDeviceSpikeHistoryToHost( AllSpikingNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info ) 
{
        int numNeurons = sim_info->totalNeurons;
        uint64_t* pSpikeHistory[numNeurons];
        HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history, numNeurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );

        int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
        for (int i = 0; i < numNeurons; i++) {
                HANDLE_ERROR( cudaMemcpy ( spike_history[i], pSpikeHistory[i],
                        max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllSpikingNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingNeurons::copyDeviceSpikeCountsToHost( AllSpikingNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info ) 
{
        int numNeurons = sim_info->totalNeurons;

        HANDLE_ERROR( cudaMemcpy ( spikeCount, allNeurons.spikeCount, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( spikeCountOffset, allNeurons.spikeCountOffset, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Clear the spike counts out of all neurons in device memory.
 *  (helper function of clearNeuronSpikeCounts)
 *
 *  @param  allNeurons         Reference to the allNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingNeurons::clearDeviceSpikeCounts( AllSpikingNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info ) 
{
        int numNeurons = sim_info->totalNeurons;

        HANDLE_ERROR( cudaMemset( allNeurons.spikeCount, 0, numNeurons * sizeof( int ) ) );
        HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeCountOffset, spikeCountOffset, numNeurons * sizeof( int ), cudaMemcpyHostToDevice ) );
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
