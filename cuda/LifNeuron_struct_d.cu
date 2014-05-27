/*
 * LifNeuron_struct_d.cu
 *
 */

#include "LIFGPUModel.h"

void LIFGPUModel::allocNeuronDeviceStruct( int count, int max_spikes ) {
	AllNeurons allNeurons;

	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C1, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C2, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Cm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.I0, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Iinject, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Inoise, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Isyn, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Rm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Tau, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Trefract, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vinit, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vreset, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vrest, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vthresh, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.hasFired, count * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.nStepsInRefr, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.neuron_type_map, count * sizeof( neuronType ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spikeCount, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.starter_map, count * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.summation_map, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spike_history, count * sizeof( uint64_t* ) ) );
	
	uint64_t* pSpikeHistory[count];
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &pSpikeHistory[i], max_spikes * sizeof( uint64_t ) ) );
	}
	HANDLE_ERROR( cudaMemcpy ( allNeurons.spike_history, pSpikeHistory,
		count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeuronsDevice, sizeof( AllNeurons ) ) );
	HANDLE_ERROR( cudaMemcpy( allNeuronsDevice, &allNeurons, sizeof( AllNeurons ), cudaMemcpyHostToDevice ) );
}

void LIFGPUModel::deleteNeuronDeviceStruct( int count ) {
	AllNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllNeurons ), cudaMemcpyDeviceToHost ) );

	uint64_t* pSpikeHistory[count];
	HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history,
		count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaFree( pSpikeHistory[i] ) );
	}

	HANDLE_ERROR( cudaFree( allNeurons.C1 ) );
	HANDLE_ERROR( cudaFree( allNeurons.C2 ) );
	HANDLE_ERROR( cudaFree( allNeurons.Cm ) );
	HANDLE_ERROR( cudaFree( allNeurons.I0 ) );
	HANDLE_ERROR( cudaFree( allNeurons.Iinject ) );
	HANDLE_ERROR( cudaFree( allNeurons.Inoise ) );
	HANDLE_ERROR( cudaFree( allNeurons.Isyn ) );
	HANDLE_ERROR( cudaFree( allNeurons.Rm ) );
	HANDLE_ERROR( cudaFree( allNeurons.Tau ) );
	HANDLE_ERROR( cudaFree( allNeurons.Trefract ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vinit ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vm ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vreset ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vrest ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vthresh ) );
	HANDLE_ERROR( cudaFree( allNeurons.hasFired ) );
	HANDLE_ERROR( cudaFree( allNeurons.nStepsInRefr ) );
	HANDLE_ERROR( cudaFree( allNeurons.neuron_type_map) );
	HANDLE_ERROR( cudaFree( allNeurons.spikeCount ) );
	HANDLE_ERROR( cudaFree( allNeurons.starter_map ) );
	HANDLE_ERROR( cudaFree( allNeurons.summation_map ) );
	HANDLE_ERROR( cudaFree( allNeurons.spike_history ) );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

void LIFGPUModel::copyNeuronHostToDevice( const AllNeurons& allNeuronsHost, int count ) { 
	AllNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllNeurons ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allNeurons.C1, allNeuronsHost.C1, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.C2, allNeuronsHost.C2, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Cm, allNeuronsHost.Cm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.I0, allNeuronsHost.I0, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Iinject, allNeuronsHost.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Inoise, allNeuronsHost.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Isyn, allNeuronsHost.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Rm, allNeuronsHost.Rm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Tau, allNeuronsHost.Tau, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Trefract, allNeuronsHost.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vinit, allNeuronsHost.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vm, allNeuronsHost.Vm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vreset, allNeuronsHost.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vrest, allNeuronsHost.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vthresh, allNeuronsHost.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.hasFired, allNeuronsHost.hasFired, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.nStepsInRefr, allNeuronsHost.nStepsInRefr, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.neuron_type_map, allNeuronsHost.neuron_type_map, count * sizeof( neuronType ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeCount, allNeuronsHost.spikeCount, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.starter_map, allNeuronsHost.starter_map, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	//HANDLE_ERROR( cudaMemcpy ( allNeurons.spike_history, allNeuronsHost.spike_history, count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );
}

void LIFGPUModel::copyNeuronDeviceToHost( AllNeurons& allNeuronsHost, int count ) {
	AllNeurons allNeurons;
	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllNeurons ), cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.C1, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.C2, allNeurons.C2, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Cm, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.I0, allNeurons.I0, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Iinject, allNeurons.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Inoise, allNeurons.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Isyn, allNeurons.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Rm, allNeurons.Rm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Tau, allNeurons.Tau, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Trefract, allNeurons.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Vinit, allNeurons.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Vm, allNeurons.Vm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Vreset, allNeurons.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Vrest, allNeurons.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.Vthresh, allNeurons.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.hasFired, allNeurons.hasFired, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.nStepsInRefr, allNeurons.nStepsInRefr, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.neuron_type_map, allNeurons.neuron_type_map, count * sizeof( neuronType ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.spikeCount, allNeurons.spikeCount, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.starter_map, allNeurons.starter_map, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.spike_history, allNeurons.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
}
