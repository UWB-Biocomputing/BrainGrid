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
 *  @param  allSynapsesProperties  Reference to the AllSTDPSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	allocSynapseDeviceStruct( allSynapsesProperties, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron, clr_info->clusterID );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSTDPSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSTDPSynapses::allocSynapseDeviceStruct( void** allSynapsesDeviceProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
	AllSTDPSynapsesProperties allSynapsesProperties;

	allocDeviceStruct( allSynapsesProperties, num_neurons, maxSynapsesPerNeuron, clusterID );

	checkCudaErrors( cudaMalloc( allSynapsesDeviceProperties, sizeof( AllSTDPSynapsesProperties ) ) );
	checkCudaErrors( cudaMemcpy ( *allSynapsesDeviceProperties, &allSynapsesProperties, sizeof( AllSTDPSynapsesProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesProperties     Reference to the AllSTDPSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSTDPSynapses::allocDeviceStruct( AllSTDPSynapsesProperties &allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
        AllSpikingSynapses::allocDeviceStruct( allSynapsesProperties, num_neurons, maxSynapsesPerNeuron, clusterID );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.total_delayPost, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.tauspost, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.tauspre, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.taupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.tauneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.STDPgap, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.Wex, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.Aneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.Apos, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.mupos, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.muneg, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.useFroemkeDanSTDP, max_total_synapses * sizeof( bool ) ) );

        // create a EventQueue objet in device memory and set the pointer in device
        postSpikeQueue->createEventQueueInDevice(&allSynapsesProperties.postSpikeQueue);
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllSTDPSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSTDPSynapses::deleteSynapseDeviceStruct( void* allSynapsesDeviceProperties ) {
	AllSTDPSynapsesProperties allSynapsesProperties;

	checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSTDPSynapsesProperties ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allSynapsesProperties );

	checkCudaErrors( cudaFree( allSynapsesDeviceProperties ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesProperties  Reference to the AllSTDPSynapsesProperties struct 
 *                             on device memory.
 */
void AllSTDPSynapses::deleteDeviceStruct( AllSTDPSynapsesProperties& allSynapsesProperties ) {
        checkCudaErrors( cudaFree( allSynapsesProperties.total_delayPost ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.tauspost ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.tauspre ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.taupos ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.tauneg ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.STDPgap ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.Wex ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.Aneg ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.Apos ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.mupos ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.muneg ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.useFroemkeDanSTDP ) );

        // delete EventQueue object in device memory.
        EventQueue::deleteEventQueueInDevice(allSynapsesProperties.postSpikeQueue);

        AllSpikingSynapses::deleteDeviceStruct( allSynapsesProperties );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesProperties     Reference to the AllSTDPSynapsesProperties struct 
 *                                on device memory.
 *  @param  sim_info              SimulationInfo to refer from.
 *  @param  clr_info              ClusterInfo to refer from.
 */
void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { // copy everything necessary
	copySynapseHostToDevice( allSynapsesProperties, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSTDPSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copySynapseHostToDevice( void* allSynapsesDeviceProperties, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
	AllSTDPSynapsesProperties allSynapsesProperties;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSTDPSynapsesProperties ), cudaMemcpyDeviceToHost ) );

	copyHostToDevice( allSynapsesDeviceProperties, allSynapsesProperties, num_neurons, maxSynapsesPerNeuron );	
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSTDPSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapses::copyHostToDevice( void* allSynapsesDeviceProperties, AllSTDPSynapsesProperties& allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        AllSpikingSynapses::copyHostToDevice( allSynapsesDeviceProperties, allSynapsesProperties, num_neurons, maxSynapsesPerNeuron );

        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;
        
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.total_delayPost, total_delayPost,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.tauspost, tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.tauspre, tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.taupos, taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.tauneg, tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.STDPgap, STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.Wex, Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.Aneg, Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.Apos, Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.mupos, mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.muneg, muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) ); 
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.useFroemkeDanSTDP, useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) ); 

        // copy event queue data from host to device.
        postSpikeQueue->copyEventQueueHostToDevice(allSynapsesProperties.postSpikeQueue);
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllSTDPSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSTDPSynapses::copySynapseDeviceToHost( void* allSynapsesDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
	// copy everything necessary
	AllSTDPSynapsesProperties allSynapsesProperties;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSTDPSynapsesProperties ), cudaMemcpyDeviceToHost ) );

	copyDeviceToHost( allSynapsesProperties, sim_info, clr_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesProperties     Reference to the AllSTDPSynapsesProperties struct 
 *                                on device memory.
 *  @param  sim_info              SimulationInfo to refer from.
 *  @param  clr_info              ClusterInfo to refer from.
 */
void AllSTDPSynapses::copyDeviceToHost( AllSTDPSynapsesProperties& allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        AllSpikingSynapses::copyDeviceToHost( allSynapsesProperties, sim_info, clr_info ) ;

	int num_neurons = clr_info->totalClusterNeurons;
	BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMemcpy ( tauspost, allSynapsesProperties.tauspost,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( tauspre, allSynapsesProperties.tauspre,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( taupos, allSynapsesProperties.taupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( tauneg, allSynapsesProperties.tauneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( STDPgap, allSynapsesProperties.STDPgap,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( Wex, allSynapsesProperties.Wex,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( Aneg, allSynapsesProperties.Aneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( Apos, allSynapsesProperties.Apos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( mupos, allSynapsesProperties.mupos,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( muneg, allSynapsesProperties.muneg,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( useFroemkeDanSTDP, allSynapsesProperties.useFroemkeDanSTDP,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );

        // copy event queue data from device to host.
        postSpikeQueue->copyEventQueueDeviceToHost(allSynapsesProperties.postSpikeQueue);
}

/*
 *  Advance all the Synapses in the simulation.
 *  Update the state of all synapses for a time step.
 *
 *  @param  allSynapsesProperties      Reference to the AllSynapsesProperties struct 
 *                                 on device memory.
 *  @param  allNeuronsProperties       Reference to the allNeurons struct on device memory.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  sim_info               SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 *  @param  iStepOffset            Offset from the current simulation step.
 */
void AllSTDPSynapses::advanceSynapses(void* allSynapsesProperties, void* allNeuronsProperties, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    DEBUG (
    int deviceId;
    checkCudaErrors( cudaGetDevice( &deviceId ) );
    assert(deviceId == clr_info->deviceId);
    ); // end DEBUG

    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;
    // Advance synapses ------------->
    advanceSTDPSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSTDPSynapsesProperties*)allSynapsesProperties, (AllSpikingNeuronsProperties*)allNeuronsProperties, max_spikes, sim_info->width, iStepOffset );
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 *  @param  allSynapsesProperties      Reference to the AllSynapsesProperties struct
 *                                 on device memory.
 *  @param  iStep                  Simulation steps to advance.
 */
void AllSTDPSynapses::advanceSpikeQueue(void* allSynapsesProperties, int iStep)
{
    AllSpikingSynapses::advanceSpikeQueue(allSynapsesProperties, iStep);

    advanceSTDPSynapsesEventQueueDevice <<< 1, 1 >>> ( (AllSTDPSynapsesProperties*)allSynapsesProperties, iStep );
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
