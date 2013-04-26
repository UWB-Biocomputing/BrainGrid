/*
 * DynamicSpikingSynapse_struct_d.cu
 * CUDA side struct of DynamicSpikingSynapse
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/**
 * Allocate data members in the allocSynapseStruct_d.
 * @param count
 */
void allocSynapseStruct_d( int count ) {
	DynamicSpikingSynapse_struct synapse;

	if ( count > 0 ) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.inUse, count * sizeof( bool ) ) );
		HANDLE_ERROR( cudaMemset( synapse.inUse, 0, count * sizeof( bool ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.summationPoint, count * sizeof( PFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.summationCoord, count * sizeof( Coordinate ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.synapseCoord, count * sizeof( Coordinate ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.deltaT, count * sizeof( FLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.W, count * sizeof( FLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.psr, count * sizeof( FLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.decay, count * sizeof( FLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.total_delay, count * sizeof( int ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.type, count * sizeof( synapseType ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.delayQueue, count * sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.ldelayQueue, count * sizeof( int ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.tau, count * sizeof( FLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.r, count * sizeof( FLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.u, count * sizeof( FLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.lastSpike, count * sizeof( uint64_t ) ) );

		HANDLE_ERROR( cudaMemcpyToSymbol ( synapse_st_d, &synapse, sizeof( DynamicSpikingSynapse_struct ) ) );
	}
}

/**
 * Deallocate data members in the DynamicSpikingSynapse_struct_d
 */
void deleteSynapseStruct_d( ) {
	DynamicSpikingSynapse_struct synapse;

	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( DynamicSpikingSynapse_struct ) ) );

	HANDLE_ERROR( cudaFree( synapse.inUse ) );
	HANDLE_ERROR( cudaFree( synapse.summationPoint ) );
	HANDLE_ERROR( cudaFree( synapse.summationCoord ) );
	HANDLE_ERROR( cudaFree( synapse.synapseCoord ) );
	HANDLE_ERROR( cudaFree( synapse.deltaT ) );
	HANDLE_ERROR( cudaFree( synapse.W ) );
	HANDLE_ERROR( cudaFree( synapse.psr ) );
	HANDLE_ERROR( cudaFree( synapse.decay ) );
	HANDLE_ERROR( cudaFree( synapse.total_delay ) );
	HANDLE_ERROR( cudaFree( synapse.type ) );
	HANDLE_ERROR( cudaFree( synapse.delayQueue ) );
	HANDLE_ERROR( cudaFree( synapse.ldelayQueue ) );
	HANDLE_ERROR( cudaFree( synapse.tau ) );
	HANDLE_ERROR( cudaFree( synapse.r ) );
	HANDLE_ERROR( cudaFree( synapse.u ) );
	HANDLE_ERROR( cudaFree( synapse.lastSpike ) );
}

/**
 * Copy DynamicSpikingSynapse_struct data for GPU processing.
 * @param synapse_h
 * @param count
 */
void copySynapseHostToDevice( DynamicSpikingSynapse_struct& synapse_h, int count ) {
	// copy everything necessary
	DynamicSpikingSynapse_struct synapse;

	if ( count > 0 ) {
        	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( DynamicSpikingSynapse_struct ) ) );

		HANDLE_ERROR( cudaMemcpy ( synapse.inUse, synapse_h.inUse, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.summationCoord, synapse_h.summationCoord, count * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.synapseCoord, synapse_h.synapseCoord, count * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.summationPoint, synapse_h.summationPoint, count * sizeof( PFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.deltaT, synapse_h.deltaT, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.W, synapse_h.W, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.psr, synapse_h.psr, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.decay, synapse_h.decay, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.total_delay, synapse_h.total_delay, count * sizeof( int ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.type, synapse_h.type, count * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.delayQueue, synapse_h.delayQueue, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.ldelayQueue, synapse_h.ldelayQueue, count * sizeof( int ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.r, synapse_h.r, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.u, synapse_h.u, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.tau, synapse_h.tau, count * sizeof( FLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.lastSpike, synapse_h.lastSpike, count * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
	}
}

/**
 * Copy data from GPU into DynamicSpikingSynapse_struct.
 * @param synapse_h
 * @param count
 */
void copySynapseDeviceToHost( DynamicSpikingSynapse_struct& synapse_h, int count ) {
	// copy everything necessary
	DynamicSpikingSynapse_struct synapse;

	if ( count > 0 ) {
        	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( DynamicSpikingSynapse_struct ) ) );

		HANDLE_ERROR( cudaMemcpy ( synapse_h.inUse, synapse.inUse, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.summationCoord, synapse.summationCoord, count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.synapseCoord, synapse.synapseCoord, count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.deltaT, synapse.deltaT, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.W, synapse.W, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.psr, synapse.psr, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.decay, synapse.decay, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.total_delay, synapse.total_delay, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.type, synapse.type, count * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.r, synapse.r, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.u, synapse.u, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.tau, synapse.tau, count * sizeof( FLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.delayQueue, synapse.delayQueue, count * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.lastSpike, synapse.lastSpike, count * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
	}
}

void copySynapseSumCoordDeviceToHost( DynamicSpikingSynapse_struct& synapse_h, int count ) {
	// copy everything necessary
	DynamicSpikingSynapse_struct synapse;

	if ( count > 0 ) {
        	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( DynamicSpikingSynapse_struct ) ) );

		HANDLE_ERROR( cudaMemcpy ( synapse_h.inUse, synapse.inUse, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.summationCoord, synapse.summationCoord, count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
	}
}
