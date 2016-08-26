/*
 * AllSTDPSynapses_d.cu
 *
 */

#include "AllSTDPSynapses.h"
#include "AllSpikingSynapses.h"
#include "GPUSpikingModel.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
	AllSTDPSynapsesDeviceProperties allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

	HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ) ) );
	HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::allocDeviceStruct( AllSTDPSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.total_delayPost, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayQueuePost, max_total_synapses * sizeof( BGSIZE ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayIdxPost, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.ldelayQueuePost, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tauspost, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tauspre, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.taupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tauneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.STDPgap, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.Wex, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.Aneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.Apos, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.mupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.muneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.useFroemkeDanSTDP, max_total_synapses * sizeof( bool ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDevice  Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
	AllSTDPSynapsesDeviceProperties allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSTDPSynapses::deleteDeviceStruct( AllSTDPSynapsesDeviceProperties& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.total_delayPost ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayQueuePost ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayIdxPost ) );
        HANDLE_ERROR( cudaFree( allSynapses.tauspost ) );
        HANDLE_ERROR( cudaFree( allSynapses.tauspre ) );
        HANDLE_ERROR( cudaFree( allSynapses.taupos ) );
        HANDLE_ERROR( cudaFree( allSynapses.tauneg ) );
        HANDLE_ERROR( cudaFree( allSynapses.STDPgap ) );
        HANDLE_ERROR( cudaFree( allSynapses.Wex ) );
        HANDLE_ERROR( cudaFree( allSynapses.Aneg ) );
        HANDLE_ERROR( cudaFree( allSynapses.Apos ) );
        HANDLE_ERROR( cudaFree( allSynapses.mupos ) );
        HANDLE_ERROR( cudaFree( allSynapses.muneg ) );
        HANDLE_ERROR( cudaFree( allSynapses.useFroemkeDanSTDP ) );

        AllSpikingSynapses::deleteDeviceStruct( allSynapses );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllSTDPSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copyHostToDevice( void* allSynapsesDevice, AllSTDPSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;
        
        HANDLE_ERROR( cudaMemcpy ( allSynapses.total_delayPost, total_delayPost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayQueuePost, delayQueuePost,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayIdxPost, delayIdxPost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.ldelayQueuePost, ldelayQueuePost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tauspost, tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tauspre, tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.taupos, taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tauneg, tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.STDPgap, STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.Wex, Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.Aneg, Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.Apos, Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.mupos, mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.muneg, muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        HANDLE_ERROR( cudaMemcpy ( allSynapses.useFroemkeDanSTDP, useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) ); 
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
	// copy everything necessary
	AllSTDPSynapsesDeviceProperties allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copyDeviceToHost( AllSTDPSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapses, sim_info ) ;

	int num_neurons = sim_info->totalNeurons;
	BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        HANDLE_ERROR( cudaMemcpy ( delayQueuePost, allSynapses.delayQueuePost,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIdxPost, allSynapses.delayIdxPost,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( ldelayQueuePost, allSynapses.ldelayQueuePost,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspost, allSynapses.tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauspre, allSynapses.tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( taupos, allSynapses.taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tauneg, allSynapses.tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( STDPgap, allSynapses.STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Wex, allSynapses.Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Aneg, allSynapses.Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( Apos, allSynapses.Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( mupos, allSynapses.mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( muneg, allSynapses.muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( useFroemkeDanSTDP, allSynapses.useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}

/*
 *  Advance all the Synapses in the simulation.
 *  Update the state of all synapses for a time step.
 *
 *  @param  allSynapsesDevice      Reference to the AllSynapsesDeviceProperties struct 
 *                                 on device memory.
 *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  sim_info               SimulationInfo class to read information from.
 */
void AllSTDPSynapses::advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;
    // Advance synapses ------------->
    advanceSTDPSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSTDPSynapsesDeviceProperties*)allSynapsesDevice, (AllSpikingNeuronsDeviceProperties*)allNeuronsDevice, max_spikes, sim_info->width );
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
void AllSTDPSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllSTDPSynapses;

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}
