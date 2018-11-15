/*
 * AllDSSynapses_d.cu
 *
 */

#include "AllDSSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include <helper_cuda.h>

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesProperties  Reference to the AllDSSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	allocSynapseDeviceStruct( allSynapsesProperties, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron, clr_info->clusterID );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllDSSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDeviceProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
	AllDSSynapsesProperties allSynapsesProperties;

	allocDeviceStruct( allSynapsesProperties, num_neurons, maxSynapsesPerNeuron, clusterID );

	checkCudaErrors( cudaMalloc( allSynapsesDeviceProperties, sizeof( AllDSSynapsesProperties ) ) );
	checkCudaErrors( cudaMemcpy ( *allSynapsesDeviceProperties, &allSynapsesProperties, sizeof( AllDSSynapsesProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesProperties     Reference to the AllDSSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllDSSynapses::allocDeviceStruct( AllDSSynapsesProperties &allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapsesProperties, num_neurons, maxSynapsesPerNeuron, clusterID );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.lastSpike, max_total_synapses * sizeof( uint64_t ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.r, max_total_synapses * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.u, max_total_synapses * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.D, max_total_synapses * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.U, max_total_synapses * sizeof( BGFLOAT ) ) );
	checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.F, max_total_synapses * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllDSSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::deleteSynapseDeviceStruct( void* allSynapsesDeviceProperties ) {
	AllDSSynapsesProperties allSynapsesProperties;

	checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllDSSynapsesProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapsesProperties );

	checkCudaErrors( cudaFree( allSynapsesDeviceProperties ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesProperties  Reference to the AllDSSynapsesProperties struct 
 *                             on device memory.
 */
void AllDSSynapses::deleteDeviceStruct( AllDSSynapsesProperties& allSynapsesProperties ) {
        checkCudaErrors( cudaFree( allSynapsesProperties.lastSpike ) );
	checkCudaErrors( cudaFree( allSynapsesProperties.r ) );
	checkCudaErrors( cudaFree( allSynapsesProperties.u ) );
	checkCudaErrors( cudaFree( allSynapsesProperties.D ) );
	checkCudaErrors( cudaFree( allSynapsesProperties.U ) );
	checkCudaErrors( cudaFree( allSynapsesProperties.F ) );

        AllSpikingSynapses::deleteDeviceStruct( allSynapsesProperties );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesProperties  Reference to the AllDSSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesProperties, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllDSSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDeviceProperties, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDSSynapsesProperties allSynapsesProperties;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllDSSynapsesProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allSynapsesDeviceProperties, allSynapsesProperties, num_neurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllDSSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copyHostToDevice( void* allSynapsesDeviceProperties, AllDSSynapsesProperties& allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allSynapsesDeviceProperties, allSynapsesProperties, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.lastSpike, lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.r, r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.u, u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.D, D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.U, U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.F, F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllDSSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllDSSynapses::copySynapseDeviceToHost( void* allSynapsesDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	// copy everything necessary
	AllDSSynapsesProperties allSynapsesProperties;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllDSSynapsesProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapsesProperties, sim_info, clr_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesProperties     Reference to the AllDSSynapsesProperties struct 
 *                                on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllDSSynapses::copyDeviceToHost( AllDSSynapsesProperties& allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapsesProperties, sim_info, clr_info ) ;

	int num_neurons = clr_info->totalClusterNeurons;
	BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMemcpy ( lastSpike, allSynapsesProperties.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( r, allSynapsesProperties.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( u, allSynapsesProperties.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( D, allSynapsesProperties.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( U, allSynapsesProperties.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( F, allSynapsesProperties.F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

/**     
 *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
 *  The class ID will be set to classSynapses_d in device memory,
 *  and the classSynapses_d will be referred to call a device function for the
 *  particular synapse class.
 *  Because we cannot use virtual function (Polymorphism) in device functions,
 *  we use this scheme.
 *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
 *  (see issue#137).
 */
void AllDSSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllDSSynapses;

    checkCudaErrors( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

