/*
 * AllSpikingSynapses.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include <helper_cuda.h>

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesProperties  Reference to the AllSpikingSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        allocSynapseDeviceStruct( allSynapsesProperties, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron, clr_info->clusterID );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSpikingSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDeviceProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
        AllSpikingSynapsesProperties allSynapsesProperties;

        allocDeviceStruct( allSynapsesProperties, num_neurons, maxSynapsesPerNeuron, clusterID );

        checkCudaErrors( cudaMalloc( allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ) ) );
        checkCudaErrors( cudaMemcpy ( *allSynapsesDeviceProperties, &allSynapsesProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesProperties     Reference to the AllSpikingSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSpikingSynapses::allocDeviceStruct( AllSpikingSynapsesProperties &allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.sourceNeuronLayoutIndex, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.destNeuronLayoutIndex, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.W, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.type, max_total_synapses * sizeof( synapseType ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.psr, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.in_use, max_total_synapses * sizeof( bool ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.synapse_counts, num_neurons * sizeof( BGSIZE ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.decay, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.tau, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.total_delay, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProperties.summation, max_total_synapses * sizeof( BGFLOAT ) ) );

        // create an EventQueue objet in device memory and set the pointer in device
        preSpikeQueue->createEventQueueInDevice(&allSynapsesProperties.preSpikeQueue);
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllSpikingSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingSynapses::deleteSynapseDeviceStruct( void* allSynapsesDeviceProperties ) {
        AllSpikingSynapsesProperties allSynapsesProperties;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyDeviceToHost ) );

        deleteDeviceStruct( allSynapsesProperties );

        checkCudaErrors( cudaFree( allSynapsesDeviceProperties ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesProperties  Reference to the AllSpikingSynapsesProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::deleteDeviceStruct( AllSpikingSynapsesProperties& allSynapsesProperties ) {
        checkCudaErrors( cudaFree( allSynapsesProperties.sourceNeuronLayoutIndex ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.destNeuronLayoutIndex ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.W ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.type ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.psr ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.in_use ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.synapse_counts ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.decay ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.tau ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.total_delay ) );
        checkCudaErrors( cudaFree( allSynapsesProperties.summation ) );

        // delete EventQueue object in device memory.
        EventQueue::deleteEventQueueInDevice(allSynapsesProperties.preSpikeQueue);

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapsesProperties.count_neurons = 0;
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesProperties  Reference to the AllSpikingSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { // copy everything necessary
        copySynapseHostToDevice( allSynapsesProperties, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSpikingSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDeviceProperties, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
        AllSpikingSynapsesProperties allSynapsesProperties;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyDeviceToHost ) );

        copyHostToDevice( allSynapsesDeviceProperties, allSynapsesProperties, num_neurons, maxSynapsesPerNeuron );
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSpikingSynapsesProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copyHostToDevice( void* allSynapsesDeviceProperties, AllSpikingSynapsesProperties& allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        allSynapsesProperties.maxSynapsesPerNeuron = maxSynapsesPerNeuron;
        allSynapsesProperties.total_synapse_counts = total_synapse_counts;
        allSynapsesProperties.count_neurons = count_neurons;
        checkCudaErrors( cudaMemcpy ( allSynapsesDeviceProperties, &allSynapsesProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyHostToDevice ) );

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapsesProperties.count_neurons = 0;

        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.sourceNeuronLayoutIndex, sourceNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.destNeuronLayoutIndex, destNeuronLayoutIndex,
                max_total_synapses * sizeof( int ),  cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.W, W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.type, type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.psr, psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.in_use, in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.synapse_counts, synapse_counts,
                        num_neurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.decay, decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.tau, tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapsesProperties.total_delay, total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );

        // copy event queue data from host to device.
        preSpikeQueue->copyEventQueueHostToDevice(allSynapsesProperties.preSpikeQueue);
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllSpikingSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copySynapseDeviceToHost( void* allSynapsesDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        // copy everything necessary
        AllSpikingSynapsesProperties allSynapsesProperties;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyDeviceToHost ) );

        copyDeviceToHost( allSynapsesProperties, sim_info, clr_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesProperties     Reference to the AllSpikingSynapsesProperties struct 
 *                                on device memory.
 *  @param  sim_info              SimulationInfo to refer from.
 *  @param  clr_info              ClusterInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceToHost( AllSpikingSynapsesProperties& allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        int num_neurons = clr_info->totalClusterNeurons;
        BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapsesProperties.synapse_counts,
                num_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuron = allSynapsesProperties.maxSynapsesPerNeuron;
        total_synapse_counts = allSynapsesProperties.total_synapse_counts;
        count_neurons = allSynapsesProperties.count_neurons;

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapsesProperties.count_neurons = 0;

        checkCudaErrors( cudaMemcpy ( sourceNeuronLayoutIndex, allSynapsesProperties.sourceNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( destNeuronLayoutIndex, allSynapsesProperties.destNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( W, allSynapsesProperties.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( type, allSynapsesProperties.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( psr, allSynapsesProperties.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( in_use, allSynapsesProperties.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( decay, allSynapsesProperties.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( tau, allSynapsesProperties.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( total_delay, allSynapsesProperties.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );

        // copy event queue data from device to host.
        preSpikeQueue->copyEventQueueDeviceToHost(allSynapsesProperties.preSpikeQueue);
}

/*
 *  Get synapse_counts in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllSpikingSynapsesProperties struct 
 *                             on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceSynapseCountsToHost(void* allSynapsesDeviceProperties, const ClusterInfo *clr_info)
{
        AllSpikingSynapsesProperties allSynapsesProperties;
        int neuron_count = clr_info->totalClusterNeurons;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapsesProperties.synapse_counts, neuron_count * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapsesProperties.count_neurons = 0;
}

/* 
 *  Get sourceNeuronLayoutIndex and in_use in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDeviceProperties  Reference to the AllSpikingSynapsesProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceSourceNeuronIdxToHost(void* allSynapsesDeviceProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
        AllSpikingSynapsesProperties allSynapsesProperties;
        BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * clr_info->totalClusterNeurons;

        checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( sourceNeuronLayoutIndex, allSynapsesProperties.sourceNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( in_use, allSynapsesProperties.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
       
        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapsesProperties.count_neurons = 0;
}

/*
 *  Set some parameters used for advanceSynapsesDevice.
 */
void AllSpikingSynapses::setAdvanceSynapsesDeviceParams()
{
    setSynapseClassID();
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
void AllSpikingSynapses::setSynapseClassID()
{
    enumClassSynapses classSynapses_h = classAllSpikingSynapses;

    checkCudaErrors( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
}

/*
 * Process inter clusters outgoing spikes.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSpikingSynapsesProperties struct
 *                                on device memory.
 */
void AllSpikingSynapses::processInterClustesOutgoingSpikes(void* allSynapsesDeviceProperties)
{
    // copy everything necessary from device to host
    AllSpikingSynapsesProperties allSynapsesProperties;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyDeviceToHost ) );

    // process inter clusters outgoing spikes 
    preSpikeQueue->processInterClustersOutgoingEvents(allSynapsesProperties.preSpikeQueue);
}

/*
 * Process inter clusters incoming spikes.
 *
 *  @param  allSynapsesDeviceProperties     Reference to the AllSpikingSynapsesProperties struct
 *                                on device memory.
 */
void AllSpikingSynapses::processInterClustesIncomingSpikes(void* allSynapsesDeviceProperties)
{
    // copy everything necessary from host to device
    AllSpikingSynapsesProperties allSynapsesProperties;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProperties, allSynapsesDeviceProperties, sizeof( AllSpikingSynapsesProperties ), cudaMemcpyDeviceToHost ) );

    // process inter clusters incoming spikes 
    preSpikeQueue->processInterClustersIncomingEvents(allSynapsesProperties.preSpikeQueue);
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
void AllSpikingSynapses::advanceSynapses(void* allSynapsesProperties, void* allNeuronsProperties, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    if (total_synapse_counts == 0)
        return;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSpikingSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSpikingSynapsesProperties*)allSynapsesProperties, iStepOffset );
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 *  @param  allSynapsesProperties      Reference to the AllSynapsesProperties struct 
 *                                 on device memory.
 *  @param  iStep                  Simulation steps to advance.
 */
void AllSpikingSynapses::advanceSpikeQueue(void* allSynapsesProperties, int iStep)
{
    advanceSpikingSynapsesEventQueueDevice <<< 1, 1 >>> ((AllSpikingSynapsesProperties*)allSynapsesProperties, iStep);
}
