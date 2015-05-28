/*
 * AllDSSynapses_d.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "Book.h"

__global__ void getFpPrePostSpikeHitDevice(void (**fpPreSpikeHit_d)(const uint32_t, AllSpikingSynapses*), void (**fpPostSpikeHit_d)(const uint32_t, AllSpikingSynapses*));

__device__ void preSpikeHitDevice( const uint32_t iSyn, AllSpikingSynapses* allSynapsesDevice );
__device__ void postSpikeHitDevice( const uint32_t iSyn, AllSpikingSynapses* allSynapsesDevice );

void AllSpikingSynapses::getFpPrePostSpikeHit(unsigned long long& fpPreSpikeHit_h, unsigned long long& fpPostSpikeHit_h)
{
    unsigned long long *fpPreSpikeHit_d, *fpPostSpikeHit_d;

    HANDLE_ERROR( cudaMalloc(&fpPreSpikeHit_d, sizeof(unsigned long long)) );
    HANDLE_ERROR( cudaMalloc(&fpPostSpikeHit_d, sizeof(unsigned long long)) );

    getFpPrePostSpikeHitDevice<<<1,1>>>((void (**)(const uint32_t, AllSpikingSynapses*))fpPreSpikeHit_d, (void (**)(const uint32_t, AllSpikingSynapses*))fpPostSpikeHit_d);

    HANDLE_ERROR( cudaMemcpy(&fpPreSpikeHit_h, fpPreSpikeHit_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(&fpPostSpikeHit_h, fpPostSpikeHit_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree( fpPreSpikeHit_d ) );
    HANDLE_ERROR( cudaFree( fpPostSpikeHit_d ) );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

__global__ void getFpPrePostSpikeHitDevice(void (**fpPreSpikeHit_d)(const uint32_t, AllSpikingSynapses*), void (**fpPostSpikeHit_d)(const uint32_t, AllSpikingSynapses*))
{
    *fpPreSpikeHit_d = preSpikeHitDevice;
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
