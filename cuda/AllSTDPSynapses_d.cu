/*
 * AllSTDPSynapses_d.cu
 *
 */

#include "AllSTDPSynapses.h"
#include "GPUSpikingModel.h"
#include "Book.h"

void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
	AllSTDPSynapses allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllSTDPSynapses ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllSTDPSynapses ), cudaMemcpyHostToDevice ) );
}

void AllSTDPSynapses::allocDeviceStruct( AllSTDPSynapses &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
        uint32_t max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.destNeuronIndex, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.W, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.summationPoint, max_total_synapses * sizeof( BGFLOAT* ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.sourceNeuronIndex, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.psr, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.decay, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.total_delay, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayQueue, max_total_synapses * sizeof( uint32_t ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayIdx, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.ldelayQueue, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.type, max_total_synapses * sizeof( synapseType ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tau, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.in_use, max_total_synapses * sizeof( bool ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.synapse_counts, num_neurons * sizeof( size_t ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.total_delayPost, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayQueuePost, max_total_synapses * sizeof( uint32_t ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayIdxPost, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.ldelayQueuePost, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tauspost, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tauspre, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.taupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tauneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.STDPgap, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.Wex, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.Aneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.Apos, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.mupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.muneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.useFroemkeDanSTDP, max_total_synapses * sizeof( bool ) ) );
}

void AllSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllSTDPSynapses allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

void AllSTDPSynapses::deleteDeviceStruct( AllSTDPSynapses& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.destNeuronIndex ) );
        HANDLE_ERROR( cudaFree( allSynapses.W ) );
        HANDLE_ERROR( cudaFree( allSynapses.summationPoint ) );
        HANDLE_ERROR( cudaFree( allSynapses.sourceNeuronIndex ) );
        HANDLE_ERROR( cudaFree( allSynapses.psr ) );
        HANDLE_ERROR( cudaFree( allSynapses.decay ) );
        HANDLE_ERROR( cudaFree( allSynapses.total_delay ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayQueue ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayIdx ) );
        HANDLE_ERROR( cudaFree( allSynapses.ldelayQueue ) );
        HANDLE_ERROR( cudaFree( allSynapses.type ) );
        HANDLE_ERROR( cudaFree( allSynapses.tau ) );
        HANDLE_ERROR( cudaFree( allSynapses.in_use ) );
        HANDLE_ERROR( cudaFree( allSynapses.synapse_counts ) );
        HANDLE_ERROR( cudaFree( allSynapses.total_delayPost ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayQueuePost ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayIdxPost ) );
        HANDLE_ERROR( cudaFree( allSynapses.tauspost ) );
        HANDLE_ERROR( cudaFree( allSynapses.tauspre ) );
        HANDLE_ERROR( cudaFree( allSynapses.taupos ) );
        HANDLE_ERROR( cudaFree( allSynapses.tauneg ) );
        HANDLE_ERROR( cudaFree( allSynapses.STDPgap ) );
        HANDLE_ERROR( cudaFree( allSynapses.Wex ) );
        HANDLE_ERROR( cudaFree( allSynapses.Aneg ) );
        HANDLE_ERROR( cudaFree( allSynapses.Apos ) );
        HANDLE_ERROR( cudaFree( allSynapses.mupos ) );
        HANDLE_ERROR( cudaFree( allSynapses.muneg ) );
        HANDLE_ERROR( cudaFree( allSynapses.useFroemkeDanSTDP ) );
}

void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );	
}

void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllSTDPSynapses allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );	
}

void AllSTDPSynapses::copyHostToDevice( void* allSynapsesDevice, AllSTDPSynapses& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        uint32_t max_total_synapses = maxSynapsesPerNeuron * num_neurons;
        
        HANDLE_ERROR( cudaMemcpy ( allSynapses.synapse_counts, synapse_counts,
                        num_neurons * sizeof( size_t ), cudaMemcpyHostToDevice ) );
        allSynapses.maxSynapsesPerNeuron = maxSynapsesPerNeuron;
        allSynapses.total_synapse_counts = total_synapse_counts;
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapses ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.destNeuronIndex, destNeuronIndex, 
                max_total_synapses * sizeof( int ),  cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.W, W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.sourceNeuronIndex, sourceNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.psr, psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.decay, decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.total_delay, total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayQueue, delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayIdx, delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.ldelayQueue, ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.type, type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tau, tau, 
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.in_use, in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.total_delayPost, total_delayPost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayQueuePost, delayQueuePost,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayIdxPost, delayIdxPost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.ldelayQueuePost, ldelayQueuePost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tauspost, tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tauspre, tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.taupos, taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tauneg, tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.STDPgap, STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.Wex, Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.Aneg, Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.Apos, Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.mupos, mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.muneg, muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.useFroemkeDanSTDP, useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) ); 
}

void AllSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllSTDPSynapses allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapses ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info );
}

void AllSTDPSynapses::copyDeviceToHost( AllSTDPSynapses& allSynapses, const SimulationInfo *sim_info ) {
	int num_neurons = sim_info->totalNeurons;
	uint32_t max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts,
                num_neurons * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuron = allSynapses.maxSynapsesPerNeuron;
        total_synapse_counts = allSynapses.total_synapse_counts;

        HANDLE_ERROR( cudaMemcpy ( destNeuronIndex, allSynapses.destNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( W, allSynapses.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndex, allSynapses.sourceNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psr, allSynapses.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( decay, allSynapses.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( total_delay, allSynapses.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueue, allSynapses.delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIdx, allSynapses.delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( ldelayQueue, allSynapses.ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( type, allSynapses.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tau, allSynapses.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( in_use, allSynapses.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( total_delayPost, allSynapses.total_delayPost,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueuePost, allSynapses.delayQueuePost,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIdxPost, allSynapses.delayIdxPost,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( ldelayQueuePost, allSynapses.ldelayQueuePost,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspost, allSynapses.tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspre, allSynapses.tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( taupos, allSynapses.taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauneg, allSynapses.tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( STDPgap, allSynapses.STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Wex, allSynapses.Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Aneg, allSynapses.Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Apos, allSynapses.Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( mupos, allSynapses.mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( muneg, allSynapses.muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( useFroemkeDanSTDP, allSynapses.useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}

/**
 *  Get synapse_counts in AllSynapses struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllSTDPSynapses::copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const SimulationInfo *sim_info)
{
        AllSTDPSynapses allSynapses;
        int neuron_count = sim_info->totalNeurons;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapses ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts, neuron_count * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
}

/** 
 *  Get summationCoord and in_use in AllSynapses struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllSTDPSynapses::copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice, const SimulationInfo *sim_info)
{
        AllSTDPSynapses allSynapses;
	uint32_t max_total_synapses = sim_info->maxSynapsesPerNeuron * sim_info->totalNeurons;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapses ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndex, allSynapses.destNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( in_use, allSynapses.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}

void AllSTDPSynapses::getFpCreateSynapse(unsigned long long& fpCreateSynapse_h)
{
    unsigned long long *fpCreateSynapse_d;

    HANDLE_ERROR( cudaMalloc(&fpCreateSynapse_d, sizeof(unsigned long long)) );

    getFpCreateSynapseDevice<<<1,1>>>((void (**)(AllSTDPSynapses*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType))fpCreateSynapse_d);

    HANDLE_ERROR( cudaMemcpy(&fpCreateSynapse_h, fpCreateSynapse_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaFree( fpCreateSynapse_d ) );
}

/** *  Advance all the Synapses in the simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllSTDPSynapses::advanceSynapses(AllSynapses* allSynapsesDevice, AllNeurons* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info)
{
    unsigned long long fpChangePSR_h;
    getFpChangePSR(fpChangePSR_h);

    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;
    // Advance synapses ------------->
    advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSTDPSynapses*)allSynapsesDevice, (void (*)(AllSTDPSynapses*, const uint32_t, const uint64_t, const BGFLOAT))fpChangePSR_h, (AllSpikingNeurons*)allNeuronsDevice, max_spikes, sim_info->width );
}

void AllSTDPSynapses::getFpPostSpikeHit(unsigned long long& fpPostSpikeHit_h)
{
    unsigned long long *fpPostSpikeHit_d;

    HANDLE_ERROR( cudaMalloc(&fpPostSpikeHit_d, sizeof(unsigned long long)) );

    getFpPostSpikeHitDevice<<<1,1>>>((void (**)(const uint32_t, AllSTDPSynapses*))fpPostSpikeHit_d);

    HANDLE_ERROR( cudaMemcpy(&fpPostSpikeHit_h, fpPostSpikeHit_d, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaFree( fpPostSpikeHit_d ) );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

__global__ void getFpCreateSynapseDevice(void (**fpCreateSynapse_d)(AllSTDPSynapses*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType))
{
    *fpCreateSynapse_d = createSynapse;
}

/**
 *  Create a Synapse and connect it to the model.
 *  @param allSynapsesDevice    Pointer to the Synapse structures in device memory.
 *  @param neuron_index         Index of the source neuron.
 *  @param synapse_index        Index of the Synapse to create.
 *  @param source_x             X location of source.
 *  @param source_y             Y location of source.
 *  @param dest_x               X location of destination.
 *  @param dest_y               Y location of destination.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createSynapse(AllSTDPSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    size_t max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    uint32_t iSyn = max_synapses * neuron_index + synapse_index;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->summationPoint[iSyn] = sum_point;
    allSynapsesDevice->destNeuronIndex[iSyn] = dest_index;
    allSynapsesDevice->sourceNeuronIndex[iSyn] = source_index;
    allSynapsesDevice->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesDevice->delayQueue[iSyn] = 0;
    allSynapsesDevice->delayIdx[iSyn] = 0;
    allSynapsesDevice->ldelayQueue[iSyn] = LENGTH_OF_DELAYQUEUE;

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

    size_t size = allSynapsesDevice->total_delay[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );

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
}

/**
* @param[in] total_synapse_counts       Total number of synapses.
* @param[in] synapseIndexMap            Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
* @param[in] simulationStep             The current simulation step.
* @param[in] deltaT                     Inner simulation step duration.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void advanceSynapsesDevice ( int total_synapse_counts, SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSTDPSynapses* allSynapsesDevice, void (*fpChangePSR)(AllSTDPSynapses*, const uint32_t, const uint64_t, const BGFLOAT), AllSpikingNeurons* allNeuronsDevice, int max_spikes, int width ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= total_synapse_counts )
            return;

    uint32_t iSyn = synapseIndexMapDevice->activeSynapseIndex[idx];

    BGFLOAT &decay = allSynapsesDevice->decay[iSyn];
    BGFLOAT &psr = allSynapsesDevice->psr[iSyn];

    // is an input in the queue?
    bool fPre = isSpikeQueueDevice(allSynapsesDevice, iSyn);
    bool fPost = isSpikeQueuePostDevice(allSynapsesDevice, iSyn);
    if (fPre || fPost) {
        BGFLOAT &tauspre = allSynapsesDevice->tauspre[iSyn];
        BGFLOAT &tauspost = allSynapsesDevice->tauspost[iSyn];
        BGFLOAT &taupos = allSynapsesDevice->taupos[iSyn];
        BGFLOAT &tauneg = allSynapsesDevice->tauneg[iSyn];
        int &total_delay = allSynapsesDevice->total_delay[iSyn];
        bool &useFroemkeDanSTDP = allSynapsesDevice->useFroemkeDanSTDP[iSyn];

        // pre and post neurons index
        int idxPre = allSynapsesDevice->sourceNeuronIndex[iSyn];
        int idxPost = allSynapsesDevice->destNeuronIndex[iSyn];
        int64_t spikeHistory, spikeHistory2;
        BGFLOAT delta;
        BGFLOAT epre, epost;

        if (fPre) {     // preSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time 
            // just one before the last spike.
            spikeHistory = getSpikeHistoryDevice(allNeuronsDevice, idxPre, -2, max_spikes);
            if (spikeHistory > 0 && useFroemkeDanSTDP) {
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
                spikeHistory = getSpikeHistoryDevice(allNeuronsDevice, idxPost, offIndex, max_spikes);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between pre-post spikes
                delta = (spikeHistory - (int64_t)simulationStep) * deltaT;

                DEBUG_SYNAPSE(
                    printf("advanceSynapsesDevice: fPre\n");
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
                    spikeHistory2 = getSpikeHistoryDevice(allNeuronsDevice, idxPost, offIndex-1, max_spikes);
                    if (spikeHistory2 == ULONG_MAX)
                        break;
                    epost = 1.0 - exp(-((spikeHistory - spikeHistory2) * deltaT) / tauspost);
                } else {
                    epost = 1.0;
                }
                stdpLearningDevice(allSynapsesDevice, iSyn, delta, epost, epre);
                --offIndex;
            }

            fpChangePSR(allSynapsesDevice, iSyn, simulationStep, deltaT);
        }

        if (fPost) {    // postSpikeHit
            // spikeCount points to the next available position of spike_history,
            // so the getSpikeHistory w/offset = -2 will return the spike time
            // just one before the last spike.
            spikeHistory = getSpikeHistoryDevice(allNeuronsDevice, idxPost, -2, max_spikes);
            if (spikeHistory > 0 && useFroemkeDanSTDP) {
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
                spikeHistory = getSpikeHistoryDevice(allNeuronsDevice, idxPre, offIndex, max_spikes);
                if (spikeHistory == ULONG_MAX)
                    break;
                // delta is the spike interval between post-pre spikes
                delta = ((int64_t)simulationStep - spikeHistory - total_delay) * deltaT;

                DEBUG_SYNAPSE(
                    printf("advanceSynapsesDevice: fPost\n");
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
                    spikeHistory2 = getSpikeHistoryDevice(allNeuronsDevice, idxPre, offIndex-1, max_spikes);
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

__device__ void stdpLearningDevice(AllSTDPSynapses* allSynapsesDevice, const uint32_t iSyn, double delta, double epost, double epre)
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

__device__ bool isSpikeQueuePostDevice(AllSTDPSynapses* allSynapsesDevice, uint32_t iSyn)
{
    uint32_t &delay_queue = allSynapsesDevice->delayQueuePost[iSyn];
    int &delayIdx = allSynapsesDevice->delayIdxPost[iSyn];
    int ldelayQueue = allSynapsesDevice->ldelayQueuePost[iSyn];

    uint32_t delayMask = (0x1 << delayIdx);
    bool isFired = delay_queue & (delayMask);
    delay_queue &= ~(delayMask);
    if ( ++delayIdx >= ldelayQueue ) {
            delayIdx = 0;
    }

    return isFired;
}

__device__ uint64_t getSpikeHistoryDevice(AllSpikingNeurons* allNeuronsDevice, int index, int offIndex, int max_spikes)
{
    // offIndex is a minus offset
    int idxSp = (allNeuronsDevice->spikeCount[index] + allNeuronsDevice->spikeCountOffset[index] +  max_spikes + offIndex) % max_spikes;
    return allNeuronsDevice->spike_history[index][idxSp];
}

__global__ void getFpPostSpikeHitDevice(void (**fpPostSpikeHit_d)(const uint32_t, AllSTDPSynapses*))
{
    *fpPostSpikeHit_d = postSpikeHitDevice;
}

__device__ void postSpikeHitDevice( const uint32_t iSyn, AllSTDPSynapses* allSynapsesDevice ) {
        uint32_t &delay_queue = allSynapsesDevice->delayQueuePost[iSyn];
        int delayIdx = allSynapsesDevice->delayIdxPost[iSyn];
        int ldelayQueue = allSynapsesDevice->ldelayQueuePost[iSyn];
        int total_delay = allSynapsesDevice->total_delayPost[iSyn];

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
