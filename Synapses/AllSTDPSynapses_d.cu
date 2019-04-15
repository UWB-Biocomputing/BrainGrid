/*
 * AllSTDPSynapses_d.cu
 *
 */

#include "AllSTDPSynapses.h"
#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include <helper_cuda.h>

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	allocSynapseDeviceStruct( allSynapsesDevice, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron, clr_info->clusterID );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
	AllSTDPSynapsesDeviceProperties allSynapses;

	allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron, clusterID );

	checkCudaErrors( cudaMalloc( allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ) ) );
	checkCudaErrors( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
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
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSTDPSynapses::allocDeviceStruct( AllSTDPSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron, clusterID );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.total_delayPost, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.tauspost, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.tauspre, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.taupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.tauneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.STDPgap, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.Wex, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.Aneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.Apos, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.mupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.muneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.useFroemkeDanSTDP, max_total_synapses * sizeof( bool ) ) );

        // create a EventQueue objet in device memory and set the pointer in device
        postSpikeQueue->createEventQueueInDevice(&allSynapses.postSpikeQueue);
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

	checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapses );

	checkCudaErrors( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSTDPSynapses::deleteDeviceStruct( AllSTDPSynapsesDeviceProperties& allSynapses ) {
        checkCudaErrors( cudaFree( allSynapses.total_delayPost ) );
        checkCudaErrors( cudaFree( allSynapses.tauspost ) );
        checkCudaErrors( cudaFree( allSynapses.tauspre ) );
        checkCudaErrors( cudaFree( allSynapses.taupos ) );
        checkCudaErrors( cudaFree( allSynapses.tauneg ) );
        checkCudaErrors( cudaFree( allSynapses.STDPgap ) );
        checkCudaErrors( cudaFree( allSynapses.Wex ) );
        checkCudaErrors( cudaFree( allSynapses.Aneg ) );
        checkCudaErrors( cudaFree( allSynapses.Apos ) );
        checkCudaErrors( cudaFree( allSynapses.mupos ) );
        checkCudaErrors( cudaFree( allSynapses.muneg ) );
        checkCudaErrors( cudaFree( allSynapses.useFroemkeDanSTDP ) );

        // delete EventQueue object in device memory.
        EventQueue::deleteEventQueueInDevice(allSynapses.postSpikeQueue);

        AllSpikingSynapses::deleteDeviceStruct( allSynapses );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  sim_info              SimulationInfo to refer from.
 *  @param  clr_info              ClusterInfo to refer from.
 */
void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesDevice, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );	
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

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

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
        
        checkCudaErrors( cudaMemcpy ( allSynapses.total_delayPost, total_delayPost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.tauspost, tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.tauspre, tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.taupos, taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.tauneg, tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.STDPgap, STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.Wex, Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.Aneg, Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.Apos, Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.mupos, mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.muneg, muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapses.useFroemkeDanSTDP, useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) ); 

        // copy event queue data from host to device.
        postSpikeQueue->copyEventQueueHostToDevice(allSynapses.postSpikeQueue);
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	// copy everything necessary
	AllSTDPSynapsesDeviceProperties allSynapses;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSTDPSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapses, sim_info, clr_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     Reference to the AllSTDPSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  sim_info              SimulationInfo to refer from.
 *  @param  clr_info              ClusterInfo to refer from.
 */
void AllSTDPSynapses::copyDeviceToHost( AllSTDPSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapses, sim_info, clr_info ) ;

	int num_neurons = clr_info->totalClusterNeurons;
	BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMemcpy ( tauspost, allSynapses.tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( tauspre, allSynapses.tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( taupos, allSynapses.taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( tauneg, allSynapses.tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( STDPgap, allSynapses.STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( Wex, allSynapses.Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( Aneg, allSynapses.Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( Apos, allSynapses.Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( mupos, allSynapses.mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( muneg, allSynapses.muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( useFroemkeDanSTDP, allSynapses.useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );

        // copy event queue data from device to host.
        postSpikeQueue->copyEventQueueDeviceToHost(allSynapses.postSpikeQueue);
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
 *  @param  clr_info               ClusterInfo to refer from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllSTDPSynapses::advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    DEBUG (
    int deviceId;
    checkCudaErrors( cudaGetDevice( &deviceId ) );
    assert(deviceId == clr_info->deviceId);
    ); // end DEBUG

    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    int blocksPerGrid = ( total_synapse_counts + sim_info->threadsPerBlock - 1 ) / sim_info->threadsPerBlock;
    // Advance synapses ------------->
    advanceSTDPSynapsesDevice <<< blocksPerGrid, sim_info->threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSTDPSynapsesDeviceProperties*)allSynapsesDevice, (AllSpikingNeuronsDeviceProperties*)allNeuronsDevice, max_spikes, sim_info->width, iStepOffset );
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 *  @param  allSynapsesDevice      Reference to the AllSynapsesDeviceProperties struct
 *                                 on device memory.
 *  @param  iStep                  Simulation steps to advance.
 */
void AllSTDPSynapses::advanceSpikeQueue(void* allSynapsesDevice, int iStep)
{
    AllSpikingSynapses::advanceSpikeQueue(allSynapsesDevice, iStep);

    advanceSTDPSynapsesEventQueueDevice <<< 1, 1 >>> ( (AllSTDPSynapsesDeviceProperties*)allSynapsesDevice, iStep );
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

    checkCudaErrors( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}
