/*
 * AllIFNeurons_d.cu
 *
 */

#include "AllIFNeurons.h"
#include "Book.h"

void AllIFNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) {
	AllIFNeurons allNeurons;

	allocDeviceStruct( allNeurons, sim_info );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIFNeurons ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIFNeurons ), cudaMemcpyHostToDevice ) );
}

void AllIFNeurons::allocDeviceStruct( AllIFNeurons &allNeurons, SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;
	int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
 
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

	// get device summation point address and set it to sim info
	sim_info->pSummationMap = allNeurons.summation_map;
}

void AllIFNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIFNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons, sim_info );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

void AllIFNeurons::deleteDeviceStruct( AllIFNeurons& allNeurons, const SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;

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
}

void AllIFNeurons::copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) { 
	AllIFNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons, sim_info );
}

void AllIFNeurons::copyHostToDevice( AllIFNeurons& allNeurons, const SimulationInfo *sim_info ) { 
	int count = sim_info->totalNeurons;

	HANDLE_ERROR( cudaMemcpy ( allNeurons.C1, C1, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.C2, C2, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Cm, Cm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.I0, I0, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Iinject, Iinject, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Inoise, Inoise, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Isyn, Isyn, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Rm, Rm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Tau, Tau, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Trefract, Trefract, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vinit, Vinit, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vm, Vm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vreset, Vreset, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vrest, Vrest, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Vthresh, Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.hasFired, hasFired, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.nStepsInRefr, nStepsInRefr, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.neuron_type_map, neuron_type_map, count * sizeof( neuronType ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeCount, spikeCount, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.starter_map, starter_map, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	//HANDLE_ERROR( cudaMemcpy ( allNeurons.spike_history, spike_history, count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );
}

void AllIFNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIFNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons, sim_info );
}

void AllIFNeurons::copyDeviceToHost( AllIFNeurons& allNeurons, const SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;

	HANDLE_ERROR( cudaMemcpy ( C1, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C2, allNeurons.C2, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cm, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( I0, allNeurons.I0, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Iinject, allNeurons.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Inoise, allNeurons.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Isyn, allNeurons.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Rm, allNeurons.Rm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Tau, allNeurons.Tau, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Trefract, allNeurons.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vinit, allNeurons.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vm, allNeurons.Vm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vreset, allNeurons.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vrest, allNeurons.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vthresh, allNeurons.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( hasFired, allNeurons.hasFired, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( nStepsInRefr, allNeurons.nStepsInRefr, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_type_map, allNeurons.neuron_type_map, count * sizeof( neuronType ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCount, allNeurons.spikeCount, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( starter_map, allNeurons.starter_map, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	//HANDLE_ERROR( cudaMemcpy ( spike_history, allNeurons.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
}
