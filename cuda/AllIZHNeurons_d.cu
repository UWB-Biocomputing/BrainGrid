/*
 * AllIZHNeurons.cu
 *
 */

#include "AllIZHNeurons.h"
#include "Book.h"

void AllIZHNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	allocDeviceStruct( allNeurons, sim_info );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIZHNeurons ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIZHNeurons ), cudaMemcpyHostToDevice ) );
}

void AllIZHNeurons::allocDeviceStruct( AllIZHNeurons &allNeurons, SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;
	int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);

	AllIFNeurons::allocDeviceStruct( allNeurons, sim_info );
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Aconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Bconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Cconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Dconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.u, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C3, count * sizeof( BGFLOAT ) ) );
}

void AllIZHNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons, sim_info );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

void AllIZHNeurons::deleteDeviceStruct( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;

	HANDLE_ERROR( cudaFree( allNeurons.Aconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Bconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Cconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Dconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.u ) );
	HANDLE_ERROR( cudaFree( allNeurons.C3 ) );

	AllIFNeurons::deleteDeviceStruct( allNeurons, sim_info );
}

void AllIZHNeurons::copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) { 
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons, sim_info );
}

void AllIZHNeurons::copyHostToDevice( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info ) { 
	int count = sim_info->totalNeurons;

	AllIFNeurons::copyHostToDevice( allNeurons, sim_info );

	HANDLE_ERROR( cudaMemcpy ( allNeurons.Aconst, Aconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Bconst, Bconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Cconst, Cconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Dconst, Dconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.u, u, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.C3, C3, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

void AllIZHNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons, sim_info );
}

void AllIZHNeurons::copyDeviceToHost( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;

	AllIFNeurons::copyDeviceToHost( allNeurons, sim_info );

	HANDLE_ERROR( cudaMemcpy ( Aconst, allNeurons.Aconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Bconst, allNeurons.Bconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cconst, allNeurons.Cconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Dconst, allNeurons.Dconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( u, allNeurons.u, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C3, allNeurons.C3, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

/**
 *  Get spike history in AllIZHNeurons struct on device memory.
 *  @param  allNeuronsDevice      Reference to the allNeurons struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons, sim_info );
}

/**
 *  Get spikeCount in AllIZHNeurons struct on device memory.
 *  @param  allNeuronsDevice      Reference to the allNeurons struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons, sim_info );
}

/** 
*  Clear the spike counts out of all Neurons.
 *  @param  allNeuronsDevice      Reference to the allNeurons struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
*/
void AllIZHNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeurons, sim_info );
}

