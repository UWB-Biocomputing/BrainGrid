/*
 * AllIFNeurons_d.cu
 *
 */

#include "AllIFNeurons.h"
#include <helper_cuda.h>

/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDevice   Reference to the AllIFNeuronsDeviceProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	AllIFNeuronsDeviceProperties allNeurons;

	allocDeviceStruct( allNeurons, sim_info, clr_info );

        checkCudaErrors( cudaMalloc( allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ) ) );
        checkCudaErrors( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::allocDeviceStruct( AllIFNeuronsDeviceProperties &allNeurons, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;
	int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
 
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.C1, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.C2, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Cm, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.I0, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Iinject, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Inoise, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Isyn, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Rm, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Tau, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Trefract, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Vinit, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Vm, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Vreset, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Vrest, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.Vthresh, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.hasFired, count * sizeof( bool ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.nStepsInRefr, count * sizeof( int ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.spikeCount, count * sizeof( int ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.spikeCountOffset, count * sizeof( int ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.summation_map, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeurons.spike_history, count * sizeof( uint64_t* ) ) );
	
	uint64_t* pSpikeHistory[count];
	for (int i = 0; i < count; i++) {
		checkCudaErrors( cudaMalloc( ( void ** ) &pSpikeHistory[i], max_spikes * sizeof( uint64_t ) ) );
	}
	checkCudaErrors( cudaMemcpy ( allNeurons.spike_history, pSpikeHistory,
		count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );

	// get device summation point address and set it to sim info
	clr_info->pClusterSummationMap = allNeurons.summation_map;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDevice   Reference to the AllIFNeuronsDeviceProperties struct on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice, const ClusterInfo *clr_info ) {
	AllIFNeuronsDeviceProperties allNeurons;

	checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons, clr_info );

	checkCudaErrors( cudaFree( allNeuronsDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::deleteDeviceStruct( AllIFNeuronsDeviceProperties& allNeurons, const ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	uint64_t* pSpikeHistory[count];
	checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history,
		count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < count; i++) {
		checkCudaErrors( cudaFree( pSpikeHistory[i] ) );
	}

	checkCudaErrors( cudaFree( allNeurons.C1 ) );
	checkCudaErrors( cudaFree( allNeurons.C2 ) );
	checkCudaErrors( cudaFree( allNeurons.Cm ) );
	checkCudaErrors( cudaFree( allNeurons.I0 ) );
	checkCudaErrors( cudaFree( allNeurons.Iinject ) );
	checkCudaErrors( cudaFree( allNeurons.Inoise ) );
	checkCudaErrors( cudaFree( allNeurons.Isyn ) );
	checkCudaErrors( cudaFree( allNeurons.Rm ) );
	checkCudaErrors( cudaFree( allNeurons.Tau ) );
	checkCudaErrors( cudaFree( allNeurons.Trefract ) );
	checkCudaErrors( cudaFree( allNeurons.Vinit ) );
	checkCudaErrors( cudaFree( allNeurons.Vm ) );
	checkCudaErrors( cudaFree( allNeurons.Vreset ) );
	checkCudaErrors( cudaFree( allNeurons.Vrest ) );
	checkCudaErrors( cudaFree( allNeurons.Vthresh ) );
	checkCudaErrors( cudaFree( allNeurons.hasFired ) );
	checkCudaErrors( cudaFree( allNeurons.nStepsInRefr ) );
	checkCudaErrors( cudaFree( allNeurons.spikeCount ) );
	checkCudaErrors( cudaFree( allNeurons.spikeCountOffset ) );
	checkCudaErrors( cudaFree( allNeurons.summation_map ) );
	checkCudaErrors( cudaFree( allNeurons.spike_history ) );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDevice   Reference to the AllIFNeuronsDeviceProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	AllIFNeuronsDeviceProperties allNeurons;

	checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyHostToDevice( AllIFNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	int count = clr_info->totalClusterNeurons;

	checkCudaErrors( cudaMemcpy ( allNeurons.C1, C1, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.C2, C2, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Cm, Cm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.I0, I0, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Iinject, Iinject, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Inoise, Inoise, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Isyn, Isyn, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Rm, Rm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Tau, Tau, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Trefract, Trefract, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Vinit, Vinit, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Vm, Vm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Vreset, Vreset, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Vrest, Vrest, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.Vthresh, Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.hasFired, hasFired, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.nStepsInRefr, nStepsInRefr, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.spikeCount, spikeCount, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeurons.spikeCountOffset, spikeCountOffset, count * sizeof( int ), cudaMemcpyHostToDevice ) );

        int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
        uint64_t* pSpikeHistory[count];
        checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                checkCudaErrors( cudaMemcpy ( pSpikeHistory[i], spike_history[i], max_spikes * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        }
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIFNeuronsDeviceProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	AllIFNeuronsDeviceProperties allNeurons;

	checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeurons         Reference to the AllIFNeuronsDeviceProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyDeviceToHost( AllIFNeuronsDeviceProperties& allNeurons, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	checkCudaErrors( cudaMemcpy ( C1, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( C2, allNeurons.C2, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Cm, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( I0, allNeurons.I0, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Iinject, allNeurons.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Inoise, allNeurons.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Isyn, allNeurons.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Rm, allNeurons.Rm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Tau, allNeurons.Tau, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Trefract, allNeurons.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vinit, allNeurons.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vm, allNeurons.Vm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vreset, allNeurons.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vrest, allNeurons.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vthresh, allNeurons.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( hasFired, allNeurons.hasFired, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( nStepsInRefr, allNeurons.nStepsInRefr, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( spikeCount, allNeurons.spikeCount, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( spikeCountOffset, allNeurons.spikeCountOffset, count * sizeof( int ), cudaMemcpyDeviceToHost ) );

        int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
        uint64_t* pSpikeHistory[count];
        checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                checkCudaErrors( cudaMemcpy ( spike_history[i], pSpikeHistory[i], max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIFNeuronsDeviceProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{        
        AllIFNeuronsDeviceProperties allNeurons;
        checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );        
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons, sim_info, clr_info );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the AllIFNeuronsDeviceProperties struct on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const ClusterInfo *clr_info )
{
        AllIFNeuronsDeviceProperties allNeurons;
        checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons, clr_info );
}

/*
 *  Clear the spike counts out of all neurons.
 *
 *  @param  allNeuronsDevice   Reference to the AllIFNeuronsDeviceProperties struct on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice, const ClusterInfo *clr_info )
{
        AllIFNeuronsDeviceProperties allNeurons;
        checkCudaErrors( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeuronsDeviceProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeurons, clr_info );
}

/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsDevice       Reference to the AllIFNeuronsDeviceProperties struct 
 *                                 on device memory.
 *  @param  allSynapsesDevice      Reference to the allSynapsesDeviceProperties struct 
 *                                 on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 */
void AllIFNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info )
{
}
