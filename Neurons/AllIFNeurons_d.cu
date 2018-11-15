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
 *  @param  allNeuronsDeviceProperties   Reference to the AllIFNeuronsProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::allocNeuronDeviceStruct( void** allNeuronsDeviceProperties, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	AllIFNeuronsProperties allNeuronsProperties;

	allocDeviceStruct( allNeuronsProperties, sim_info, clr_info );

        checkCudaErrors( cudaMalloc( allNeuronsDeviceProperties, sizeof( AllIFNeuronsProperties ) ) );
        checkCudaErrors( cudaMemcpy ( *allNeuronsDeviceProperties, &allNeuronsProperties, sizeof( AllIFNeuronsProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeuronsProperties         Reference to the AllIFNeuronsProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::allocDeviceStruct( AllIFNeuronsProperties &allNeuronsProperties, SimulationInfo *sim_info, ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;
	int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
 
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.C1, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.C2, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Cm, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.I0, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Iinject, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Inoise, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Isyn, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Rm, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Tau, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Trefract, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vinit, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vm, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vreset, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vrest, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.Vthresh, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.hasFired, count * sizeof( bool ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.nStepsInRefr, count * sizeof( int ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.spikeCount, count * sizeof( int ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.spikeCountOffset, count * sizeof( int ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.summation_map, count * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.spike_history, count * sizeof( uint64_t* ) ) );
	
	uint64_t* pSpikeHistory[count];
	for (int i = 0; i < count; i++) {
		checkCudaErrors( cudaMalloc( ( void ** ) &pSpikeHistory[i], max_spikes * sizeof( uint64_t ) ) );
	}
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.spike_history, pSpikeHistory,
		count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );

	// get device summation point address and set it to sim info
	clr_info->pClusterSummationMap = allNeuronsProperties.summation_map;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIFNeuronsProperties struct on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::deleteNeuronDeviceStruct( void* allNeuronsDeviceProperties, const ClusterInfo *clr_info ) {
	AllIFNeuronsProperties allNeuronsProperties;

	checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIFNeuronsProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeuronsProperties, clr_info );

	checkCudaErrors( cudaFree( allNeuronsDeviceProperties ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeuronsProperties         Reference to the AllIFNeuronsProperties struct.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::deleteDeviceStruct( AllIFNeuronsProperties& allNeuronsProperties, const ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	uint64_t* pSpikeHistory[count];
	checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProperties.spike_history,
		count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < count; i++) {
		checkCudaErrors( cudaFree( pSpikeHistory[i] ) );
	}

	checkCudaErrors( cudaFree( allNeuronsProperties.C1 ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.C2 ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Cm ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.I0 ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Iinject ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Inoise ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Isyn ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Rm ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Tau ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Trefract ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Vinit ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Vm ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Vreset ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Vrest ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.Vthresh ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.hasFired ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.nStepsInRefr ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.spikeCount ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.spikeCountOffset ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.summation_map ) );
	checkCudaErrors( cudaFree( allNeuronsProperties.spike_history ) );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIFNeuronsProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyNeuronHostToDevice( void* allNeuronsDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	AllIFNeuronsProperties allNeuronsProperties;

	checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIFNeuronsProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeuronsProperties, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeuronsProperties         Reference to the AllIFNeuronsProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyHostToDevice( AllIFNeuronsProperties& allNeuronsProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { 
	int count = clr_info->totalClusterNeurons;

	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.C1, C1, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.C2, C2, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Cm, Cm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.I0, I0, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Iinject, Iinject, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Inoise, Inoise, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Isyn, Isyn, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Rm, Rm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Tau, Tau, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Trefract, Trefract, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Vinit, Vinit, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Vm, Vm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Vreset, Vreset, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Vrest, Vrest, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.Vthresh, Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.hasFired, hasFired, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.nStepsInRefr, nStepsInRefr, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.spikeCount, spikeCount, count * sizeof( int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy ( allNeuronsProperties.spikeCountOffset, spikeCountOffset, count * sizeof( int ), cudaMemcpyHostToDevice ) );

        int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
        uint64_t* pSpikeHistory[count];
        checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProperties.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                checkCudaErrors( cudaMemcpy ( pSpikeHistory[i], spike_history[i], max_spikes * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        }
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIFNeuronsProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceToHost( void* allNeuronsDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	AllIFNeuronsProperties allNeuronsProperties;

	checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIFNeuronsProperties ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeuronsProperties, sim_info, clr_info );
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeuronsProperties         Reference to the AllIFNeuronsProperties struct.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyDeviceToHost( AllIFNeuronsProperties& allNeuronsProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	int count = clr_info->totalClusterNeurons;

	checkCudaErrors( cudaMemcpy ( C1, allNeuronsProperties.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( C2, allNeuronsProperties.C2, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Cm, allNeuronsProperties.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( I0, allNeuronsProperties.I0, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Iinject, allNeuronsProperties.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Inoise, allNeuronsProperties.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Isyn, allNeuronsProperties.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Rm, allNeuronsProperties.Rm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Tau, allNeuronsProperties.Tau, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Trefract, allNeuronsProperties.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vinit, allNeuronsProperties.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vm, allNeuronsProperties.Vm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vreset, allNeuronsProperties.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vrest, allNeuronsProperties.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( Vthresh, allNeuronsProperties.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( hasFired, allNeuronsProperties.hasFired, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( nStepsInRefr, allNeuronsProperties.nStepsInRefr, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( spikeCount, allNeuronsProperties.spikeCount, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaMemcpy ( spikeCountOffset, allNeuronsProperties.spikeCountOffset, count * sizeof( int ), cudaMemcpyDeviceToHost ) );

        int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
        uint64_t* pSpikeHistory[count];
        checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProperties.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
        for (int i = 0; i < count; i++) {
                checkCudaErrors( cudaMemcpy ( spike_history[i], pSpikeHistory[i], max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        }
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIFNeuronsProperties struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{        
        // Set device ID
        checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

        AllIFNeuronsProperties allNeuronsProperties;
        checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIFNeuronsProperties ), cudaMemcpyDeviceToHost ) );        
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeuronsProperties, sim_info, clr_info );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDeviceProperties   Reference to the AllIFNeuronsProperties struct on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDeviceProperties, const ClusterInfo *clr_info )
{
        // Set device ID
        checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

        AllIFNeuronsProperties allNeuronsProperties;
        checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, allNeuronsDeviceProperties, sizeof( AllIFNeuronsProperties ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeuronsProperties, clr_info );
}

/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsProperties       Reference to the AllIFNeuronsProperties struct 
 *                                 on device memory.
 *  @param  allSynapsesProperties      Reference to the allSynapsesProperties struct 
 *                                 on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllIFNeurons::advanceNeurons( IAllSynapses &synapses, void* allNeuronsProperties, void* allSynapsesProperties, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset )
{
}
