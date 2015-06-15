/*
 * AllSpikingSynapses.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "Book.h"

//! Perform updating synapses for one time step.
__global__ void advanceSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapses* allSynapsesDevice, void (*fpChangePSR)(AllSpikingSynapses*, const uint32_t, const uint64_t, const BGFLOAT) );

__device__ bool isSpikeQueueDevice(AllSpikingSynapses* allSynapsesDevice, uint32_t iSyn);

__global__ void getFpPreSpikeHitDevice(void (**fpPreSpikeHit_d)(const uint32_t, AllSpikingSynapses*));

__global__ void getFpPostSpikeHitDevice(void (**fpPostSpikeHit_d)(const uint32_t, AllSpikingSynapses*));

__device__ void preSpikeHitDevice( const uint32_t iSyn, AllSpikingSynapses* allSynapsesDevice );

__device__ void postSpikeHitDevice( const uint32_t iSyn, AllSpikingSynapses* allSynapsesDevice );

__global__ void getFpChangePSRDevice(void (**fpChangePSR_d)(AllSpikingSynapses*, const uint32_t, const uint64_t, const BGFLOAT));

__device__ void changePSR(AllSpikingSynapses* allSynapsesDevice, const uint32_t, const uint64_t, const BGFLOAT deltaT);

/**
 *  Advance all the Synapses in the simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllSpikingSynapses::advanceSynapses(AllSynapses* allSynapsesDevice, AllNeurons* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info)
{
    unsigned long long fpChangePSR_h;
    getFpChangePSR(fpChangePSR_h);

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSpikingSynapses*)allSynapsesDevice, (void (*)(AllSpikingSynapses*, const uint32_t, const uint64_t, const BGFLOAT))fpChangePSR_h );
}

void AllSpikingSynapses::getFpPreSpikeHit(unsigned long long& fpPreSpikeHit_h)
{
    unsigned long long *fpPreSpikeHit_d;

    HANDLE_ERROR( cudaMalloc(&fpPreSpikeHit_d, sizeof(unsigned long long)) );

    getFpPreSpikeHitDevice<<<1,1>>>((void (**)(const uint32_t, AllSpikingSynapses*))fpPreSpikeHit_d);

    HANDLE_ERROR( cudaMemcpy(&fpPreSpikeHit_h, fpPreSpikeHit_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree( fpPreSpikeHit_d ) );
}

void AllSpikingSynapses::getFpPostSpikeHit(unsigned long long& fpPostSpikeHit_h)
{
    unsigned long long *fpPostSpikeHit_d;

    HANDLE_ERROR( cudaMalloc(&fpPostSpikeHit_d, sizeof(unsigned long long)) );

    getFpPostSpikeHitDevice<<<1,1>>>((void (**)(const uint32_t, AllSpikingSynapses*))fpPostSpikeHit_d);

    HANDLE_ERROR( cudaMemcpy(&fpPostSpikeHit_h, fpPostSpikeHit_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree( fpPostSpikeHit_d ) );
}

void AllSpikingSynapses::getFpChangePSR(unsigned long long& fpChangePSR_h)
{
    unsigned long long *fpChangePSR_d;

    HANDLE_ERROR( cudaMalloc(&fpChangePSR_d, sizeof(unsigned long long)) );

    getFpChangePSRDevice<<<1,1>>>((void (**)(AllSpikingSynapses*, const uint32_t, const uint64_t, const BGFLOAT))fpChangePSR_d);

    HANDLE_ERROR( cudaMemcpy(&fpChangePSR_h, fpChangePSR_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaFree( fpChangePSR_d ) );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

/** 
* @param[in] total_synapse_counts       Total number of synapses.
* @param[in] synapseIndexMap            Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
* @param[in] simulationStep             The current simulation step.
* @param[in] deltaT                     Inner simulation step duration.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void advanceSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSpikingSynapses* allSynapsesDevice, void (*fpChangePSR)(AllSpikingSynapses*, const uint32_t, const uint64_t, const BGFLOAT) ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= total_synapse_counts )
                return;

        uint32_t iSyn = synapseIndexMapDevice->activeSynapseIndex[idx];

        BGFLOAT &psr = allSynapsesDevice->psr[iSyn];
        BGFLOAT decay = allSynapsesDevice->decay[iSyn];

        // Checks if there is an input spike in the queue.
        bool isFired = isSpikeQueueDevice(allSynapsesDevice, iSyn);

        // is an input in the queue?
        if (isFired) {
                fpChangePSR(allSynapsesDevice, iSyn, simulationStep, deltaT);
        }
        // decay the post spike response
        psr *= decay;
}

__device__ bool isSpikeQueueDevice(AllSpikingSynapses* allSynapsesDevice, uint32_t iSyn)
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

__global__ void getFpPreSpikeHitDevice(void (**fpPreSpikeHit_d)(const uint32_t, AllSpikingSynapses*))
{
    *fpPreSpikeHit_d = preSpikeHitDevice;
}

__global__ void getFpPostSpikeHitDevice(void (**fpPostSpikeHit_d)(const uint32_t, AllSpikingSynapses*))
{
    *fpPostSpikeHit_d = postSpikeHitDevice;
}

__device__ void preSpikeHitDevice( const uint32_t iSyn, AllSpikingSynapses* allSynapsesDevice ) {
        uint32_t &delay_queue = allSynapsesDevice->delayQueue[iSyn];
        int delayIdx = allSynapsesDevice->delayIdx[iSyn];
        int ldelayQueue = allSynapsesDevice->ldelayQueue[iSyn];
        int total_delay = allSynapsesDevice->total_delay[iSyn];

        // Add to spike queue

        // calculate index where to insert the spike into delayQueue
        int idx = delayIdx +  total_delay;
        if ( idx >= ldelayQueue ) {
                idx -= ldelayQueue;
        }

        // set a spike
        //assert( !(delay_queue[0] & (0x1 << idx)) );
        delay_queue |= (0x1 << idx);
}

__device__ void postSpikeHitDevice( const uint32_t iSyn, AllSpikingSynapses* allSynapsesDevice ) {
}

__global__ void getFpChangePSRDevice(void (**fpChangePSR_d)(AllSpikingSynapses*, const uint32_t, const uint64_t, const BGFLOAT))
{
    *fpChangePSR_d = changePSR;
}

__device__ void changePSR(AllSpikingSynapses* allSynapsesDevice, const uint32_t iSyn, const uint64_t simulationStep, const BGFLOAT deltaT)
{
    BGFLOAT &psr = allSynapsesDevice->psr[iSyn];
    BGFLOAT &W = allSynapsesDevice->W[iSyn];
    BGFLOAT &decay = allSynapsesDevice->decay[iSyn];

    psr += ( W / decay );    // calculate psr
}

