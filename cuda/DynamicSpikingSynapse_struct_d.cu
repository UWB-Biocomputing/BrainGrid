/*
 * DynamicSpikingSynapse_struct_d.cu
 *
 */

#include "LIFGPUModel.h"

__global__ void setSynapseSummationPointDevice(int num_neurons, AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, int width);

void LIFGPUModel::allocSynapseDeviceStruct( int num_neurons, int max_synapses ) {
	AllSynapses allSynapses_0;
	AllSynapses allSynapses_1(num_neurons, 0);

	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.summationCoord, num_neurons * sizeof( Coordinate* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.W, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.summationPoint, num_neurons * sizeof( BGFLOAT** ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.synapseCoord, num_neurons * sizeof( Coordinate* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.psr, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.decay, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.total_delay, num_neurons * sizeof( int* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.delayQueue, num_neurons * sizeof( uint32_t** ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.delayIdx, num_neurons * sizeof( int* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.ldelayQueue, num_neurons * sizeof( int* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.type, num_neurons * sizeof( synapseType* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.tau, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.r, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.u, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.D, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.U, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.F, num_neurons * sizeof( BGFLOAT* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.lastSpike, num_neurons * sizeof( uint64_t* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.in_use, num_neurons * sizeof( bool* ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_0.synapse_counts, num_neurons * sizeof( size_t ) ) );

	for (int i = 0; i < num_neurons; i++) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.summationCoord[i], max_synapses * sizeof( Coordinate ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.W[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.summationPoint[i], max_synapses * sizeof( BGFLOAT* ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.synapseCoord[i], max_synapses * sizeof( Coordinate ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.psr[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.decay[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.total_delay[i], max_synapses * sizeof( int ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.delayQueue[i], max_synapses * sizeof( uint32_t* ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.delayIdx[i], max_synapses * sizeof( int ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.ldelayQueue[i], max_synapses * sizeof( int ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.type[i], max_synapses * sizeof( synapseType ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.tau[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.r[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.u[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.D[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.U[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.F[i], max_synapses * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.lastSpike[i], max_synapses * sizeof( uint64_t ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses_1.in_use[i], max_synapses * sizeof( bool ) ) );

		uint32_t* pDelayQueue[max_synapses];
		for (int j = 0; j < max_synapses; j++) {
			HANDLE_ERROR( cudaMalloc( ( void ** ) &pDelayQueue[j], sizeof( uint32_t ) ) );
		}
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayQueue[i], pDelayQueue, 
			max_synapses * sizeof( uint32_t* ), cudaMemcpyHostToDevice ) );
	}
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.summationCoord, allSynapses_1.summationCoord, 
		num_neurons * sizeof( Coordinate* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.W, allSynapses_1.W, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) ); 
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.summationPoint, allSynapses_1.summationPoint, 
		num_neurons * sizeof( BGFLOAT** ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapseCoord, allSynapses_1.synapseCoord, 
		num_neurons * sizeof( Coordinate* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.psr, allSynapses_1.psr, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.decay, allSynapses_1.decay, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.total_delay, allSynapses_1.total_delay, 
		num_neurons * sizeof( int* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayQueue, allSynapses_1.delayQueue, 
		num_neurons * sizeof( uint32_t** ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.delayIdx, allSynapses_1.delayIdx, 
		num_neurons * sizeof( int* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.ldelayQueue, allSynapses_1.ldelayQueue, 
		num_neurons * sizeof( int* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.type, allSynapses_1.type, 
		num_neurons * sizeof( synapseType* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.tau, allSynapses_1.tau, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.r, allSynapses_1.r, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.u, allSynapses_1.u, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.D, allSynapses_1.D, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.U, allSynapses_1.U, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.F, allSynapses_1.F, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.lastSpike, allSynapses_1.lastSpike, 
		num_neurons * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.in_use, allSynapses_1.in_use, 
		num_neurons * sizeof( bool* ), cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapsesDevice, sizeof( AllSynapses ) ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses_0, sizeof( AllSynapses ), cudaMemcpyHostToDevice ) );
}

void LIFGPUModel::deleteSynapseDeviceStruct( int num_neurons, int max_synapses ) {
	AllSynapses allSynapses_0;
	AllSynapses allSynapses_1(num_neurons, 0);

	HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllSynapses ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationCoord, allSynapses_0.summationCoord, 
		num_neurons * sizeof( Coordinate* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.W, allSynapses_0.W, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) ); 
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationPoint, allSynapses_0.summationPoint, 
		num_neurons * sizeof( BGFLOAT** ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.synapseCoord, allSynapses_0.synapseCoord, 
		num_neurons * sizeof( Coordinate* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.psr, allSynapses_0.psr, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.decay, allSynapses_0.decay, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.total_delay, allSynapses_0.total_delay, 
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayQueue, allSynapses_0.delayQueue, 
		num_neurons * sizeof( uint32_t** ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayIdx, allSynapses_0.delayIdx, 
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.ldelayQueue, allSynapses_0.ldelayQueue, 
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.type, allSynapses_0.type, 
		num_neurons * sizeof( synapseType* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.tau, allSynapses_0.tau, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.r, allSynapses_0.r, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.u, allSynapses_0.u, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.D, allSynapses_0.D, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.U, allSynapses_0.U, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.F, allSynapses_0.F, 
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.lastSpike, allSynapses_0.lastSpike, 
		num_neurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.in_use, allSynapses_0.in_use, 
		num_neurons * sizeof( bool* ), cudaMemcpyDeviceToHost ) );

	for (int i = 0; i < num_neurons; i++) {
		uint32_t* pDelayQueue[max_synapses];
		HANDLE_ERROR( cudaMemcpy ( pDelayQueue, allSynapses_1.delayQueue[i],
			max_synapses * sizeof( uint32_t* ), cudaMemcpyDeviceToHost ) );
		for (int j = 0; j < max_synapses; j++) {
			HANDLE_ERROR( cudaFree( pDelayQueue[j] ) );
		}

		HANDLE_ERROR( cudaFree( allSynapses_1.summationCoord[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.W[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.summationPoint[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.synapseCoord[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.psr[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.decay[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.total_delay[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.delayQueue[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.delayIdx[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.ldelayQueue[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.type[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.tau[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.r[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.u[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.D[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.U[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.F[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.lastSpike[i] ) );
		HANDLE_ERROR( cudaFree( allSynapses_1.in_use[i] ) );
	}

	HANDLE_ERROR( cudaFree( allSynapses_0.summationCoord ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.W ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.summationPoint ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.synapseCoord ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.psr ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.decay ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.total_delay ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.delayQueue ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.delayIdx ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.ldelayQueue ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.type ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.tau ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.r ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.u ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.D ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.U ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.F ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.lastSpike ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.in_use ) );
	HANDLE_ERROR( cudaFree( allSynapses_0.synapse_counts ) );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

void LIFGPUModel::copySynapseHostToDevice( const AllSynapses& allSynapsesHost, const SimulationInfo *sim_info ) { // copy everything necessary
	int num_neurons = sim_info->totalNeurons;
	int max_synapses = sim_info->maxSynapsesPerNeuron;
	AllSynapses allSynapses_0;
	AllSynapses allSynapses_1(num_neurons, 0);

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllSynapses ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_0.synapse_counts, allSynapsesHost.synapse_counts, 
			num_neurons * sizeof( size_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( &allSynapsesDevice->max_synapses, &allSynapsesHost.max_synapses, 
			sizeof( size_t ), cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationCoord, allSynapses_0.summationCoord, 
		num_neurons * sizeof( Coordinate* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.W, allSynapses_0.W,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationPoint, allSynapses_0.summationPoint,
	//	num_neurons * sizeof( BGFLOAT** ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.synapseCoord, allSynapses_0.synapseCoord,
		num_neurons * sizeof( Coordinate* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.psr, allSynapses_0.psr,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.decay, allSynapses_0.decay,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.total_delay, allSynapses_0.total_delay,
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayQueue, allSynapses_0.delayQueue,
	//	num_neurons * sizeof( uint32_t** ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayIdx, allSynapses_0.delayIdx,
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.ldelayQueue, allSynapses_0.ldelayQueue,
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.type, allSynapses_0.type,
		num_neurons * sizeof( synapseType* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.tau, allSynapses_0.tau,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.r, allSynapses_0.r,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.u, allSynapses_0.u,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.D, allSynapses_0.D,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.U, allSynapses_0.U,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.F, allSynapses_0.F,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.lastSpike, allSynapses_0.lastSpike,
		num_neurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.in_use, allSynapses_0.in_use,
		num_neurons * sizeof( bool* ), cudaMemcpyDeviceToHost ) );

	for (int i = 0; i < num_neurons; i++) {
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationCoord[i], allSynapsesHost.summationCoord[i], 
			max_synapses * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.W[i], allSynapsesHost.W[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		//HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationPoint[i], allSynapsesHost.summationPoint[i],
		//	max_synapses * sizeof( BGFLOAT* ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.synapseCoord[i], allSynapsesHost.synapseCoord[i],
			max_synapses * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.psr[i], allSynapsesHost.psr[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.decay[i], allSynapsesHost.decay[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.total_delay[i], allSynapsesHost.total_delay[i],
			max_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
		//HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayQueue[i], allSynapsesHost.delayQueue[i],
		//	max_synapses * sizeof( uint32_t* ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayIdx[i], allSynapsesHost.delayIdx[i],
			max_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.ldelayQueue[i], allSynapsesHost.ldelayQueue[i],
			max_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.type[i], allSynapsesHost.type[i],
			max_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.tau[i], allSynapsesHost.tau[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.r[i], allSynapsesHost.r[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.u[i], allSynapsesHost.u[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.D[i], allSynapsesHost.D[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.U[i], allSynapsesHost.U[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.F[i], allSynapsesHost.F[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.lastSpike[i], allSynapsesHost.lastSpike[i],
			max_synapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapses_1.in_use[i], allSynapsesHost.in_use[i],
			max_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) );

                // set summation points
                const int threadsPerBlock = 256;
                int blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;       
                setSynapseSummationPointDevice <<< blocksPerGrid, threadsPerBlock >>> (sim_info->totalNeurons, allNeuronsDevice, allSynapsesDevice, sim_info->width);
	}
}

/**
 * Set the summation points in device memory
 * @param[in] num_neurons        Number of neurons.
 * @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
 * @param[in] width              Width of neuron map (assumes square).
 */
__global__ void setSynapseSummationPointDevice(int num_neurons, AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_neurons )
        return;

    int src_neuron = idx; 
    for (int syn_index = 0; syn_index < allSynapsesDevice->synapse_counts[src_neuron]; syn_index++) {
        int dest_neuron = allSynapsesDevice->summationCoord[src_neuron][syn_index].x 
            + allSynapsesDevice->summationCoord[src_neuron][syn_index].y * width;
        allSynapsesDevice->summationPoint[src_neuron][syn_index] = &( allNeuronsDevice->summation_map[dest_neuron] );
    }
}

void LIFGPUModel::copySynapseDeviceToHost( AllSynapses& allSynapsesHost, int num_neurons, int max_synapses ) {
	// copy everything necessary
	AllSynapses allSynapses_0;
	AllSynapses allSynapses_1(num_neurons, 0);

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllSynapses ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.synapse_counts, allSynapses_0.synapse_counts, 
		num_neurons * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( &allSynapsesHost.max_synapses, &allSynapsesDevice->max_synapses, 
		sizeof( size_t ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationCoord, allSynapses_0.summationCoord, 
		num_neurons * sizeof( Coordinate* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.W, allSynapses_0.W,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationPoint, allSynapses_0.summationPoint,
	//	num_neurons * sizeof( BGFLOAT** ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.synapseCoord, allSynapses_0.synapseCoord,
		num_neurons * sizeof( Coordinate* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.psr, allSynapses_0.psr,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.decay, allSynapses_0.decay,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.total_delay, allSynapses_0.total_delay,
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayQueue, allSynapses_0.delayQueue,
	//	num_neurons * sizeof( uint32_t** ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.delayIdx, allSynapses_0.delayIdx,
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.ldelayQueue, allSynapses_0.ldelayQueue,
		num_neurons * sizeof( int* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.type, allSynapses_0.type,
		num_neurons * sizeof( synapseType* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.tau, allSynapses_0.tau,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.r, allSynapses_0.r,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.u, allSynapses_0.u,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.D, allSynapses_0.D,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.U, allSynapses_0.U,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.F, allSynapses_0.F,
		num_neurons * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.lastSpike, allSynapses_0.lastSpike,
		num_neurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapses_1.in_use, allSynapses_0.in_use,
		num_neurons * sizeof( bool* ), cudaMemcpyDeviceToHost ) );

	for (int i = 0; i < num_neurons; i++) {
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.summationCoord[i], allSynapses_1.summationCoord[i],
			max_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.W[i], allSynapses_1.W[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		//HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.summationPoint[i], allSynapses_1.summationPoint[i],
		//	max_synapses * sizeof( BGFLOAT* ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.synapseCoord[i], allSynapses_1.synapseCoord[i],
			max_synapses * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.psr[i], allSynapses_1.psr[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.decay[i], allSynapses_1.decay[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.total_delay[i], allSynapses_1.total_delay[i],
			max_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
		//HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.delayQueue[i], allSynapses_1.delayQueue[i],
		//	max_synapses * sizeof( uint32_t* ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.delayIdx[i], allSynapses_1.delayIdx[i],
			max_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.ldelayQueue[i], allSynapses_1.ldelayQueue[i],
			max_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.type[i], allSynapses_1.type[i],
			max_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.tau[i], allSynapses_1.tau[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.r[i], allSynapses_1.r[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.u[i], allSynapses_1.u[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.D[i], allSynapses_1.D[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.U[i], allSynapses_1.U[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.F[i], allSynapses_1.F[i],
			max_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.lastSpike[i], allSynapses_1.lastSpike[i],
			max_synapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.in_use[i], allSynapses_1.in_use[i],
			max_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	}
}
