/*
 * AllIZHNeurons.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "AllIZHNeurons.h"
#include "AllNeuronsDeviceFuncs.h"

#include <helper_cuda.h>

/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	AllIZHNeuronsDeviceProperties allNeurons;

	allocDeviceStruct( allNeurons, sim_info, clr_info );

        checkCudaErrors( cudaMalloc( allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ) ) );
        checkCudaErrors( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::allocDeviceStruct( AllIZHNeuronsDeviceProperties &allNeurons, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	AllIFNeurons::allocDeviceStruct( allNeurons, sim_info, clr_info );
 
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Aconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Bconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Cconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Dconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.u, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.C3, count * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice, const ClusterInfo *clr_info ) {
	AllIZHNeuronsDeviceProperties allNeurons;

	checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons, clr_info );

	checkCudaErrors( cudaFree( allNeuronsDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::deleteDeviceStruct( AllIZHNeuronsDeviceProperties& allNeurons, const ClusterInfo *clr_info ) {
	checkCudaErrors( cudaFree( allNeurons.Aconst ) );
	checkCudaErrors( cudaFree( allNeurons.Bconst ) );
	checkCudaErrors( cudaFree( allNeurons.Cconst ) );
	checkCudaErrors( cudaFree( allNeurons.Dconst ) );
	checkCudaErrors( cudaFree( allNeurons.u ) );
	checkCudaErrors( cudaFree( allNeurons.C3 ) );

	AllIFNeurons::deleteDeviceStruct( allNeurons, clr_info );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	AllIZHNeuronsDeviceProperties allNeurons;

	checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyHostToDevice( AllIZHNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	int count = clr_info->totalClusterNeurons;

	AllIFNeurons::copyHostToDevice( allNeurons, sim_info, clr_info );

	checkCudaErrors( cudaMemcpy ( allNeurons.Aconst, Aconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Bconst, Bconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Cconst, Cconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Dconst, Dconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.u, u, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.C3, C3, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	AllIZHNeuronsDeviceProperties allNeurons;

	checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeurons         Reference to the AllIZHNeuronsDeviceProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyDeviceToHost( AllIZHNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	AllIFNeurons::copyDeviceToHost( allNeurons, sim_info, clr_info );

	checkCudaErrors( cudaMemcpy ( Aconst, allNeurons.Aconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Bconst, allNeurons.Bconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Cconst, allNeurons.Cconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Dconst, allNeurons.Dconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( u, allNeurons.u, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( C3, allNeurons.C3, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        AllIZHNeuronsDeviceProperties allNeurons;
        checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons, sim_info, clr_info );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIZHNeuronsDeviceProperties struct 
 *                             on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const ClusterInfo *clr_info )
{
        AllIZHNeuronsDeviceProperties allNeurons;
        checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons, clr_info );
}

/**
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
 *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  clr_info               ClusterInfo class to read information from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllIZHNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset)
{
    DEBUG (
    int deviceId;
    checkCudaErrors( cudaGetDevice( &deviceId ) );
    assert(deviceId == clr_info->deviceId);
    ); // end DEBUG

    int neuron_count = clr_info->totalClusterNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    // Advance neurons ------------->
    advanceIZHNeuronsDevice <<< clr_info->blocksPerGrid, clr_info->threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIZHNeuronsDeviceProperties *)allNeuronsDevice, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice, synapseIndexMapDevice, m_fAllowBackPropagation, iStepOffset );
}

