/*
 * AllDSSynapses_d.cu
 *
 */

#include "AllDSSynapses.h"
#include "GPUSpikingModel.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
	AllDSSynapsesDeviceProperties allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::allocDeviceStruct( AllDSSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.lastSpike, max_total_synapses * sizeof( uint64_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.r, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.u, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.D, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.U, max_total_synapses * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.F, max_total_synapses * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllDSSynapsesDeviceProperties allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllDSSynapses::deleteDeviceStruct( AllDSSynapsesDeviceProperties& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.lastSpike ) );
	HANDLE_ERROR( cudaFree( allSynapses.r ) );
	HANDLE_ERROR( cudaFree( allSynapses.u ) );
	HANDLE_ERROR( cudaFree( allSynapses.D ) );
	HANDLE_ERROR( cudaFree( allSynapses.U ) );
	HANDLE_ERROR( cudaFree( allSynapses.F ) );

        AllSpikingSynapses::deleteDeviceStruct( allSynapses );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllDSSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copyHostToDevice( void* allSynapsesDevice, AllDSSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( allSynapses.lastSpike, lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.r, r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.u, u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.D, D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.U, U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.F, F,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  Reference to the AllDSSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllDSSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllDSSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllDSSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     Reference to the AllDSSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDSSynapses::copyDeviceToHost( AllDSSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapses, sim_info ) ;

	int num_neurons = sim_info->totalNeurons;
	BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( lastSpike, allSynapses.lastSpike,
                max_total_synapses * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( r, allSynapses.r,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( u, allSynapses.u,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( D, allSynapses.D,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( U, allSynapses.U,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( F, allSynapses.F,
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

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

