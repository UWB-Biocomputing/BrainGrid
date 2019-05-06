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
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        allocSynapseDeviceStruct( allSynapsesDevice, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron, clr_info->clusterID );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
        AllSpikingSynapsesDeviceProperties allSynapses;

        allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron, clusterID );

        checkCudaErrors( cudaMalloc( allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ) ) );
        checkCudaErrors( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *  (Helper function of allocSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 *  @param  clusterID             The cluster ID of the cluster.
 */
void AllSpikingSynapses::allocDeviceStruct( AllSpikingSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID ) {
        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.sourceNeuronLayoutIndex, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.destNeuronLayoutIndex, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.W, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.type, max_total_synapses * sizeof( synapseType ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.psr, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.in_use, max_total_synapses * sizeof( bool ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.synapse_counts, num_neurons * sizeof( BGSIZE ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.decay, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.tau, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.total_delay, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.summation, max_total_synapses * sizeof( BGFLOAT ) ) );

        // create an EventQueue objet in device memory and set the pointer in device
        preSpikeQueue->createEventQueueInDevice(&allSynapses.preSpikeQueue);
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingSynapses::deleteSynapseDeviceStruct( void* allSynapsesDevice ) {
        AllSpikingSynapsesDeviceProperties allSynapses;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        deleteDeviceStruct( allSynapses );

        checkCudaErrors( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::deleteDeviceStruct( AllSpikingSynapsesDeviceProperties& allSynapses ) {
        checkCudaErrors( cudaFree( allSynapses.sourceNeuronLayoutIndex ) );
        checkCudaErrors( cudaFree( allSynapses.destNeuronLayoutIndex ) );
        checkCudaErrors( cudaFree( allSynapses.W ) );
        checkCudaErrors( cudaFree( allSynapses.type ) );
        checkCudaErrors( cudaFree( allSynapses.psr ) );
        checkCudaErrors( cudaFree( allSynapses.in_use ) );
        checkCudaErrors( cudaFree( allSynapses.synapse_counts ) );
        checkCudaErrors( cudaFree( allSynapses.decay ) );
        checkCudaErrors( cudaFree( allSynapses.tau ) );
        checkCudaErrors( cudaFree( allSynapses.total_delay ) );
        checkCudaErrors( cudaFree( allSynapses.summation ) );

        // delete EventQueue object in device memory.
        EventQueue::deleteEventQueueInDevice(allSynapses.preSpikeQueue);

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.count_neurons = 0;
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) { // copy everything necessary
        copySynapseHostToDevice( allSynapsesDevice, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary
        AllSpikingSynapsesDeviceProperties allSynapses;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        copyHostToDevice( allSynapsesDevice, allSynapses, num_neurons, maxSynapsesPerNeuron );
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDevice)
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copyHostToDevice( void* allSynapsesDevice, AllSpikingSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron ) { // copy everything necessary 
        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        allSynapses.maxSynapsesPerNeuron = maxSynapsesPerNeuron;
        allSynapses.total_synapse_counts = total_synapse_counts;
        allSynapses.count_neurons = count_neurons;
        checkCudaErrors( cudaMemcpy ( allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapses.count_neurons = 0;

        checkCudaErrors( cudaMemcpy ( allSynapses.sourceNeuronLayoutIndex, sourceNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.destNeuronLayoutIndex, destNeuronLayoutIndex,
                max_total_synapses * sizeof( int ),  cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.W, W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.type, type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.psr, psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.in_use, in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.synapse_counts, synapse_counts,
                        num_neurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.decay, decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.tau, tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.total_delay, total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );

        // copy event queue data from host to device.
        preSpikeQueue->copyEventQueueHostToDevice(allSynapses.preSpikeQueue);
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        // copy everything necessary
        AllSpikingSynapsesDeviceProperties allSynapses;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        copyDeviceToHost( allSynapses, sim_info, clr_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  sim_info              SimulationInfo to refer from.
 *  @param  clr_info              ClusterInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceToHost( AllSpikingSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) {
        int num_neurons = clr_info->totalClusterNeurons;
        BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts,
                num_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuron = allSynapses.maxSynapsesPerNeuron;
        total_synapse_counts = allSynapses.total_synapse_counts;
        count_neurons = allSynapses.count_neurons;

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapses.count_neurons = 0;

        checkCudaErrors( cudaMemcpy ( sourceNeuronLayoutIndex, allSynapses.sourceNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( destNeuronLayoutIndex, allSynapses.destNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( W, allSynapses.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( type, allSynapses.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( psr, allSynapses.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( in_use, allSynapses.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( decay, allSynapses.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( tau, allSynapses.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( total_delay, allSynapses.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );

        // copy event queue data from device to host.
        preSpikeQueue->copyEventQueueDeviceToHost(allSynapses.preSpikeQueue);
}

/*
 *  Get synapse_counts in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const ClusterInfo *clr_info)
{
        AllSpikingSynapsesDeviceProperties allSynapses;
        int neuron_count = clr_info->totalClusterNeurons;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts, neuron_count * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.count_neurons = 0;
}

/* 
 *  Get sourceNeuronLayoutIndex and in_use in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceSourceNeuronIdxToHost(void* allSynapsesDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
        AllSpikingSynapsesDeviceProperties allSynapses;
        BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * clr_info->totalClusterNeurons;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( sourceNeuronLayoutIndex, allSynapses.sourceNeuronLayoutIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( in_use, allSynapses.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
       
        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.count_neurons = 0;
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
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct
 *                                on device memory.
 */
void AllSpikingSynapses::processInterClustesOutgoingSpikes(void* allSynapsesDevice)
{
    // copy everything necessary from device to host
    AllSpikingSynapsesDeviceProperties allSynapses;

    checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

    // process inter clusters outgoing spikes 
    preSpikeQueue->processInterClustersOutgoingEvents(allSynapses.preSpikeQueue);
}

/*
 * Process inter clusters incoming spikes.
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct
 *                                on device memory.
 */
void AllSpikingSynapses::processInterClustesIncomingSpikes(void* allSynapsesDevice)
{
    // copy everything necessary from host to device
    AllSpikingSynapsesDeviceProperties allSynapses;

    checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

    // process inter clusters incoming spikes 
    preSpikeQueue->processInterClustersIncomingEvents(allSynapses.preSpikeQueue);
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
void AllSpikingSynapses::advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset)
{
    if (total_synapse_counts == 0)
        return;

    // CUDA parameters
    int blocksPerGrid = ( total_synapse_counts + clr_info->threadsPerBlock - 1 ) / clr_info->threadsPerBlock;

    // Advance synapses ------------->
    //advanceSpikingSynapsesDevice <<< blocksPerGrid, clr_info->threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice, iStepOffset);
    advanceDSSSynapsesDevice << < blocksPerGrid, clr_info->threadsPerBlock >> > (total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllDSSynapsesDeviceProperties*)allSynapsesDevice, iStepOffset);
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 *  @param  allSynapsesDevice      Reference to the AllSynapsesDeviceProperties struct 
 *                                 on device memory.
 *  @param  iStep                  Simulation steps to advance.
 */
void AllSpikingSynapses::advanceSpikeQueue(void* allSynapsesDevice, int iStep)
{
    advanceSpikingSynapsesEventQueueDevice <<< 1, 1 >>> ((AllSpikingSynapsesDeviceProperties*)allSynapsesDevice, iStep);
}
