/*
 * DynamicSpikingSynapse_struct_d.cu
 *
 */

#include "LIFGPUModel.h"

void LIFGPUModel::allocSynapseDeviceStruct( AllSynapsesDevice*& allSynapsesDevice, int num_neurons, int max_synapses ) {
	AllSynapsesDevice allSynapses;
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

	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice, sizeof( AllSynapsesDevice ) ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses, sizeof( AllSynapsesDevice ), cudaMemcpyHostToDevice ) );
}

void LIFGPUModel::deleteSynapseDeviceStruct( AllSynapsesDevice* allSynapsesDevice, int num_neurons, int max_synapses ) {
	AllSynapsesDevice allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSynapsesDevice ), cudaMemcpyDeviceToHost ) );

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

void LIFGPUModel::copySynapseHostToDevice( AllSynapsesDevice* allSynapsesDevice, const AllSynapses& allSynapsesHost, int num_neurons, int max_synapses ) { // copy everything necessary
	uint32_t max_total_synapses = max_synapses * num_neurons;
	AllSynapsesDevice allSynapses_0;
	AllSynapsesDevice allSynapses_1(num_neurons, max_synapses);

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllSynapsesDevice ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapse_counts, allSynapsesHost.synapse_counts, 
			num_neurons * sizeof( size_t ), cudaMemcpyHostToDevice ) );
	allSynapses_0.max_synapses = allSynapsesHost.max_synapses;	
	allSynapses_0.total_synapse_counts = allSynapsesHost.total_synapse_counts;	
	HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses_0, sizeof( AllSynapsesDevice ), cudaMemcpyHostToDevice ) );

	for (int i = 0; i < num_neurons; i++) {
		memcpy( &allSynapses_1.summationCoord[max_synapses * i], allSynapsesHost.summationCoord[i], 
			max_synapses * sizeof( Coordinate ) );
		memcpy ( &allSynapses_1.summationCoord[max_synapses * i], allSynapsesHost.summationCoord[i], 
			max_synapses * sizeof( Coordinate ) );
		memcpy ( &allSynapses_1.W[max_synapses * i], allSynapsesHost.W[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.synapseCoord[max_synapses * i], allSynapsesHost.synapseCoord[i],
			max_synapses * sizeof( Coordinate ) );
		memcpy ( &allSynapses_1.psr[max_synapses * i], allSynapsesHost.psr[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.decay[max_synapses * i], allSynapsesHost.decay[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.total_delay[max_synapses * i], allSynapsesHost.total_delay[i],
			max_synapses * sizeof( int ) );
		for (int j = 0; j < max_synapses; j++) {
			allSynapses_1.delayQueue[max_synapses * i + j] = *allSynapsesHost.delayQueue[i][j];
		}
		memcpy ( &allSynapses_1.delayIdx[max_synapses * i], allSynapsesHost.delayIdx[i],
			max_synapses * sizeof( int ) );
		memcpy ( &allSynapses_1.ldelayQueue[max_synapses * i], allSynapsesHost.ldelayQueue[i],
			max_synapses * sizeof( int ) );
		memcpy ( &allSynapses_1.type[max_synapses * i], allSynapsesHost.type[i],
			max_synapses * sizeof( synapseType ) );
		memcpy ( &allSynapses_1.tau[max_synapses * i], allSynapsesHost.tau[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.r[max_synapses * i], allSynapsesHost.r[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy (  &allSynapses_1.u[max_synapses * i], allSynapsesHost.u[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.D[max_synapses * i], allSynapsesHost.D[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.U[max_synapses * i], allSynapsesHost.U[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.F[max_synapses * i], allSynapsesHost.F[i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( &allSynapses_1.lastSpike[max_synapses * i], allSynapsesHost.lastSpike[i],
			max_synapses * sizeof( uint64_t ) );
		memcpy ( &allSynapses_1.in_use[max_synapses * i], allSynapsesHost.in_use[i],
			max_synapses * sizeof( bool ) );

	}

        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.summationCoord, allSynapses_1.summationCoord,
                max_total_synapses * sizeof( Coordinate ),  cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.W, allSynapses_0.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapseCoord, allSynapses_1.synapseCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.psr, allSynapses_0.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.decay, allSynapses_1.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.total_delay, allSynapses_1.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayQueue, allSynapses_1.delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayIdx, allSynapses_1.delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.ldelayQueue, allSynapses_1.ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.type, allSynapses_1.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.tau, allSynapses_1.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.r, allSynapses_1.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.u, allSynapses_1.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.D, allSynapses_1.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.U, allSynapses_1.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.F, allSynapses_1.F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.lastSpike, allSynapses_1.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_0.in_use, allSynapses_1.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
}

void LIFGPUModel::copySynapseDeviceToHost( AllSynapsesDevice* allSynapsesDevice, AllSynapses& allSynapsesHost, int num_neurons, int max_synapses ) {
	// copy everything necessary
	AllSynapsesDevice allSynapses_0;
	AllSynapsesDevice allSynapses_1(num_neurons, max_synapses);
	uint32_t max_total_synapses = max_synapses * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllSynapsesDevice ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.synapse_counts, allSynapses_0.synapse_counts, 
		num_neurons * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	allSynapsesHost.max_synapses = allSynapses_0.max_synapses;
	allSynapsesHost.total_synapse_counts = allSynapses_0.total_synapse_counts;

        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationCoord, allSynapses_0.summationCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.W, allSynapses_0.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.synapseCoord, allSynapses_0.synapseCoord,
                max_total_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.psr, allSynapses_0.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.decay, allSynapses_0.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.total_delay, allSynapses_0.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayQueue, allSynapses_0.delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayIdx, allSynapses_0.delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.ldelayQueue, allSynapses_0.ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.type, allSynapses_0.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.tau, allSynapses_0.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.r, allSynapses_0.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.u, allSynapses_0.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.D, allSynapses_0.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.U, allSynapses_0.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.F, allSynapses_0.F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.lastSpike, allSynapses_0.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.in_use, allSynapses_0.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );

	for (int i = 0; i < num_neurons; i++) {
		memcpy ( allSynapsesHost.summationCoord[i], &allSynapses_1.summationCoord[max_synapses * i],
			max_synapses * sizeof( Coordinate ) );
		memcpy ( allSynapsesHost.W[i], &allSynapses_1.W[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.synapseCoord[i], &allSynapses_1.synapseCoord[max_synapses * i],
			max_synapses * sizeof( Coordinate ) );
		memcpy ( allSynapsesHost.psr[i], &allSynapses_1.psr[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.decay[i], &allSynapses_1.decay[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.total_delay[i], &allSynapses_1.total_delay[max_synapses * i],
			max_synapses * sizeof( int ) );
		for (int j = 0; j < max_synapses; j++) {
			*allSynapsesHost.delayQueue[i][j] = allSynapses_1.delayQueue[max_synapses * i + j];
		}
		memcpy ( allSynapsesHost.delayIdx[i], &allSynapses_1.delayIdx[max_synapses * i],
			max_synapses * sizeof( int ) );
		memcpy ( allSynapsesHost.ldelayQueue[i], &allSynapses_1.ldelayQueue[max_synapses * i],
			max_synapses * sizeof( int ) );
		memcpy ( allSynapsesHost.type[i], &allSynapses_1.type[max_synapses * i],
			max_synapses * sizeof( synapseType ) );
		memcpy ( allSynapsesHost.tau[i], &allSynapses_1.tau[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.r[i], &allSynapses_1.r[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.u[i], &allSynapses_1.u[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.D[i], &allSynapses_1.D[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.U[i], &allSynapses_1.U[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.F[i], &allSynapses_1.F[max_synapses * i],
			max_synapses * sizeof( BGFLOAT ) );
		memcpy ( allSynapsesHost.lastSpike[i], &allSynapses_1.lastSpike[max_synapses * i],
			max_synapses * sizeof( uint64_t ) );
		memcpy ( allSynapsesHost.in_use[i], &allSynapses_1.in_use[max_synapses * i],
			max_synapses * sizeof( bool ) );
	}
}
