/*
 * AllDSSynapses_d.cu
 *
 */

#include "AllDSSynapses.h"
#include "Book.h"

void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	int num_neurons = sim_info->totalNeurons;
	int max_synapses = sim_info->maxSynapsesPerNeuron;

	allocSynapseDeviceStruct( allSynapsesDevice, num_neurons, max_synapses );
}

void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int max_synapses ) {
	AllDSSynapses allSynapses;
	uint32_t max_total_synapses = max_synapses * num_neurons;

	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.summationCoord, max_total_synapses * sizeof( Coordinate ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.W, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.summationPoint, max_total_synapses * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.synapseCoord, max_total_synapses * sizeof( Coordinate ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.psr, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.decay, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.total_delay, max_total_synapses * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayQueue, max_total_synapses * sizeof( uint32_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayIdx, max_total_synapses * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.ldelayQueue, max_total_synapses * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.type, max_total_synapses * sizeof( synapseType ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tau, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.r, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.u, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.D, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.U, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.F, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.lastSpike, max_total_synapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.in_use, max_total_synapses * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.synapse_counts, num_neurons * sizeof( size_t ) ) );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllDSSynapses ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDSSynapses ), cudaMemcpyHostToDevice ) );
}

void AllDSSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
        int num_neurons = sim_info->totalNeurons;
        int max_synapses = sim_info->maxSynapsesPerNeuron;

	deleteSynapseDeviceStruct( allSynapsesDevice, num_neurons, max_synapses );
}

void AllDSSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice, int num_neurons, int max_synapses ) {
	AllDSSynapses allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapses ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaFree( allSynapses.summationCoord ) );
	HANDLE_ERROR( cudaFree( allSynapses.W ) );
	HANDLE_ERROR( cudaFree( allSynapses.summationPoint ) );
	HANDLE_ERROR( cudaFree( allSynapses.synapseCoord ) );
	HANDLE_ERROR( cudaFree( allSynapses.psr ) );
	HANDLE_ERROR( cudaFree( allSynapses.decay ) );
	HANDLE_ERROR( cudaFree( allSynapses.total_delay ) );
	HANDLE_ERROR( cudaFree( allSynapses.delayQueue ) );
	HANDLE_ERROR( cudaFree( allSynapses.delayIdx ) );
	HANDLE_ERROR( cudaFree( allSynapses.ldelayQueue ) );
	HANDLE_ERROR( cudaFree( allSynapses.type ) );
	HANDLE_ERROR( cudaFree( allSynapses.tau ) );
	HANDLE_ERROR( cudaFree( allSynapses.r ) );
	HANDLE_ERROR( cudaFree( allSynapses.u ) );
	HANDLE_ERROR( cudaFree( allSynapses.D ) );
	HANDLE_ERROR( cudaFree( allSynapses.U ) );
	HANDLE_ERROR( cudaFree( allSynapses.F ) );
	HANDLE_ERROR( cudaFree( allSynapses.lastSpike ) );
	HANDLE_ERROR( cudaFree( allSynapses.in_use ) );
	HANDLE_ERROR( cudaFree( allSynapses.synapse_counts ) );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	int num_neurons = sim_info->totalNeurons;
	int max_synapses =  sim_info->maxSynapsesPerNeuron;

	copySynapseHostToDevice( allSynapsesDevice, num_neurons, max_synapses );	
}

void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int max_synapses ) { // copy everything necessary
	uint32_t max_total_synapses = max_synapses * num_neurons;
	AllDSSynapses allSynapses_0;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllDSSynapses ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapse_counts, synapse_counts, 
			num_neurons * sizeof( size_t ), cudaMemcpyHostToDevice ) );
	allSynapses_0.maxSynapsesPerNeuron = maxSynapsesPerNeuron;	
	allSynapses_0.total_synapse_counts = total_synapse_counts;	
	HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses_0, sizeof( AllDSSynapses ), cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.summationCoord, summationCoord,
                max_total_synapses * sizeof( Coordinate ),  cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.W, allSynapses_0.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapseCoord, synapseCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.psr, allSynapses_0.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.decay, decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.total_delay, total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayQueue, delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayIdx, delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.ldelayQueue, ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.type, type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.tau, tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.r, r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.u, u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.D, D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.U, U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.F, F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.lastSpike, lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.in_use, in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
}

void AllDSSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllDSSynapses allSynapses_0;
	int num_neurons = sim_info->totalNeurons;
	int max_synapses = sim_info->maxSynapsesPerNeuron;
	uint32_t max_total_synapses = max_synapses * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllDSSynapses ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( synapse_counts, allSynapses_0.synapse_counts, 
		num_neurons * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	maxSynapsesPerNeuron = allSynapses_0.maxSynapsesPerNeuron;
	total_synapse_counts = allSynapses_0.total_synapse_counts;

        HANDLE_ERROR( cudaMemcpy ( summationCoord, allSynapses_0.summationCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( W, allSynapses_0.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapseCoord, allSynapses_0.synapseCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psr, allSynapses_0.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( decay, allSynapses_0.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( total_delay, allSynapses_0.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueue, allSynapses_0.delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIdx, allSynapses_0.delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( ldelayQueue, allSynapses_0.ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( type, allSynapses_0.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tau, allSynapses_0.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( r, allSynapses_0.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( u, allSynapses_0.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( D, allSynapses_0.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( U, allSynapses_0.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( F, allSynapses_0.F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( lastSpike, allSynapses_0.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( in_use, allSynapses_0.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}
