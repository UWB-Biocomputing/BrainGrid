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
 *  @param  allNeuronsDeviceProperties   Reference to the AllIZHNeuronsProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::allocNeuronDeviceStruct( void** allNeuronsDeviceProperties, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	AllIZHNeuronsProperties allNeuronsProperties;

	allocDeviceStruct( allNeuronsProperties, sim_info, clr_info );

        checkCudaErrors( cudaMalloc( allNeuronsDeviceProperties, sizeof( AllIZHNeuronsProperties ) ) );
        checkCudaErrors( cudaMemcpy ( *allNeuronsDeviceProperties, &allNeuronsProperties, sizeof( AllIZHNeuronsProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeuronsProperties         Reference to the AllIZHNeuronsProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::allocDeviceStruct( AllIZHNeuronsProperties &allNeuronsProperties, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	AllIFNeurons::allocDeviceStruct( allNeuronsProperties, sim_info, clr_info );
 
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Aconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Bconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Cconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Dconst, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.u, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.C3, count * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIZHNeuronsProperties struct 
 *                             on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::deleteNeuronDeviceStruct( void* allNeuronsDeviceProperties, const ClusterInfo *clr_info ) {
	AllIZHNeuronsProperties allNeuronsProperties;

	checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIZHNeuronsProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeuronsProperties, clr_info );

	checkCudaErrors( cudaFree( allNeuronsDeviceProperties ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeuronsProperties         Reference to the AllIZHNeuronsProperties struct.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::deleteDeviceStruct( AllIZHNeuronsProperties& allNeuronsProperties, const ClusterInfo *clr_info ) {
	checkCudaErrors( cudaFree( allNeuronsProperties.Aconst ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Bconst ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Cconst ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Dconst ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.u ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.C3 ) );

	AllIFNeurons::deleteDeviceStruct( allNeuronsProperties, clr_info );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIZHNeuronsProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronHostToDevice( void* allNeuronsDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	AllIZHNeuronsProperties allNeuronsProperties;

	checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIZHNeuronsProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeuronsProperties, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeuronsProperties         Reference to the AllIZHNeuronsProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyHostToDevice( AllIZHNeuronsProperties& allNeuronsProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	int count = clr_info->totalClusterNeurons;

	AllIFNeurons::copyHostToDevice( allNeuronsProperties, sim_info, clr_info );

	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Aconst, Aconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Bconst, Bconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Cconst, Cconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Dconst, Dconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.u, u, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.C3, C3, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIZHNeuronsProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceToHost( void* allNeuronsDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	AllIZHNeuronsProperties allNeuronsProperties;

	checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIZHNeuronsProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeuronsProperties, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeuronsProperties         Reference to the AllIZHNeuronsProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyDeviceToHost( AllIZHNeuronsProperties& allNeuronsProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	AllIFNeurons::copyDeviceToHost( allNeuronsProperties, sim_info, clr_info );

	checkCudaErrors( cudaMemcpy ( Aconst, allNeuronsProperties.Aconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Bconst, allNeuronsProperties.Bconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Cconst, allNeuronsProperties.Cconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Dconst, allNeuronsProperties.Dconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( u, allNeuronsProperties.u, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( C3, allNeuronsProperties.C3, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIZHNeuronsProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        AllIZHNeuronsProperties allNeuronsProperties;
        checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIZHNeuronsProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeuronsProperties, sim_info, clr_info );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIZHNeuronsProperties struct 
 *                             on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDeviceProperties, const ClusterInfo *clr_info )
{
        AllIZHNeuronsProperties allNeuronsProperties;
        checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIZHNeuronsProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeuronsProperties, clr_info );
}

/**
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsProperties       Reference to the allNeurons struct on device memory.
 *  @param  allSynapsesProperties      Reference to the allSynapses struct on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  clr_info               ClusterInfo class to read information from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllIZHNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsProperties, void* allSynapsesProperties, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset)
{
    DEBUG (
    int deviceId;
    checkCudaErrors( cudaGetDevice( &deviceId ) );
    assert(deviceId == clr_info->deviceId);
    ); // end DEBUG

    int neuron_count = clr_info->totalClusterNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceIZHNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIZHNeuronsProperties *)allNeuronsProperties, (AllSpikingSynapsesProperties*)allSynapsesProperties, synapseIndexMapDevice, m_fAllowBackPropagation, iStepOffset );
}

