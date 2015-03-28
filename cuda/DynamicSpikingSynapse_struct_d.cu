/*
 * DynamicSpikingSynapse_struct_d.cu
 *
 */

#include "LIFGPUModel.h"

void LIFGPUModel::allocSynapseDeviceStruct( AllDSSynapses*& allSynapsesDevice, int num_neurons, int max_synapses ) {
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

	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice, sizeof( AllDSSynapses ) ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses, sizeof( AllDSSynapses ), cudaMemcpyHostToDevice ) );
}

void LIFGPUModel::deleteSynapseDeviceStruct( AllDSSynapses* allSynapsesDevice, int num_neurons, int max_synapses ) {
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

void LIFGPUModel::copySynapseHostToDevice( AllDSSynapses* allSynapsesDevice, const AllDSSynapses& allSynapsesHost, int num_neurons, int max_synapses ) { // copy everything necessary
	uint32_t max_total_synapses = max_synapses * num_neurons;
	AllDSSynapses allSynapses_0;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllDSSynapses ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapse_counts, allSynapsesHost.synapse_counts, 
			num_neurons * sizeof( size_t ), cudaMemcpyHostToDevice ) );
	allSynapses_0.maxSynapsesPerNeuron = allSynapsesHost.maxSynapsesPerNeuron;	
	allSynapses_0.total_synapse_counts = allSynapsesHost.total_synapse_counts;	
	HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses_0, sizeof( AllDSSynapses ), cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.summationCoord, allSynapsesHost.summationCoord,
                max_total_synapses * sizeof( Coordinate ),  cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.W, allSynapses_0.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapseCoord, allSynapsesHost.synapseCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.psr, allSynapses_0.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.decay, allSynapsesHost.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.total_delay, allSynapsesHost.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayQueue, allSynapsesHost.delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayIdx, allSynapsesHost.delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.ldelayQueue, allSynapsesHost.ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.type, allSynapsesHost.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.tau, allSynapsesHost.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.r, allSynapsesHost.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.u, allSynapsesHost.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.D, allSynapsesHost.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.U, allSynapsesHost.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.F, allSynapsesHost.F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.lastSpike, allSynapsesHost.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.in_use, allSynapsesHost.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
}

void LIFGPUModel::copySynapseDeviceToHost( AllDSSynapses* allSynapsesDevice, AllDSSynapses& allSynapsesHost, int num_neurons, int max_synapses ) {
	// copy everything necessary
	AllDSSynapses allSynapses_0;
	uint32_t max_total_synapses = max_synapses * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllDSSynapses ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.synapse_counts, allSynapses_0.synapse_counts, 
		num_neurons * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	allSynapsesHost.maxSynapsesPerNeuron = allSynapses_0.maxSynapsesPerNeuron;
	allSynapsesHost.total_synapse_counts = allSynapses_0.total_synapse_counts;

        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.summationCoord, allSynapses_0.summationCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.W, allSynapses_0.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.synapseCoord, allSynapses_0.synapseCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.psr, allSynapses_0.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.decay, allSynapses_0.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.total_delay, allSynapses_0.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.delayQueue, allSynapses_0.delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.delayIdx, allSynapses_0.delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.ldelayQueue, allSynapses_0.ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.type, allSynapses_0.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.tau, allSynapses_0.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.r, allSynapses_0.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.u, allSynapses_0.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.D, allSynapses_0.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.U, allSynapses_0.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.F, allSynapses_0.F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.lastSpike, allSynapses_0.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.in_use, allSynapses_0.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}
