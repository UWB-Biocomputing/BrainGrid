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
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info ) {
        allocSynapseDeviceStruct( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
}

/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron ) {
        AllSpikingSynapsesDeviceProperties allSynapses;

        allocDeviceStruct( allSynapses, num_neurons, maxSynapsesPerNeuron );

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
 */
void AllSpikingSynapses::allocDeviceStruct( AllSpikingSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron ) {
        BGSIZE max_total_synapses = maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.sourceNeuronIndex, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.destNeuronIndex, max_total_synapses * sizeof( int ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.W, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.type, max_total_synapses * sizeof( synapseType ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.psr, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.in_use, max_total_synapses * sizeof( bool ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.synapse_counts, num_neurons * sizeof( BGSIZE ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.decay, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.tau, max_total_synapses * sizeof( BGFLOAT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &allSynapses.total_delay, max_total_synapses * sizeof( int ) ) );

        // create a EventQueue objet in device memory and set the pointer to preSpikeQueue.
        EventQueue **pEventQueue; // temporary buffer to save pointer to EventQueue object.

        // allocate device memory for the buffer.
        checkCudaErrors( cudaMalloc( ( void ** ) &pEventQueue, sizeof( EventQueue * ) ) );

        // create a EventQueue object in device memory.
        allocEventQueueDevice <<< 1, 1 >>> ( max_total_synapses, pEventQueue );

        // save the pointer of the object.
        checkCudaErrors( cudaMemcpy ( &allSynapses.preSpikeQueue, pEventQueue, sizeof( EventQueue * ), cudaMemcpyDeviceToHost ) );

        // free device memory for the buffer.
        checkCudaErrors( cudaFree( pEventQueue ) );
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
        checkCudaErrors( cudaFree( allSynapses.sourceNeuronIndex ) );
        checkCudaErrors( cudaFree( allSynapses.destNeuronIndex ) );
        checkCudaErrors( cudaFree( allSynapses.W ) );
        checkCudaErrors( cudaFree( allSynapses.type ) );
        checkCudaErrors( cudaFree( allSynapses.psr ) );
        checkCudaErrors( cudaFree( allSynapses.in_use ) );
        checkCudaErrors( cudaFree( allSynapses.synapse_counts ) );
        checkCudaErrors( cudaFree( allSynapses.decay ) );
        checkCudaErrors( cudaFree( allSynapses.tau ) );
        checkCudaErrors( cudaFree( allSynapses.total_delay ) );

        // delete EventQueue object in device memory.
        deleteEventQueueDevice <<< 1, 1 >>> ( allSynapses.preSpikeQueue );

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
 */
void AllSpikingSynapses::copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info ) { // copy everything necessary
        copySynapseHostToDevice( allSynapsesDevice, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );
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

        checkCudaErrors( cudaMemcpy ( allSynapses.sourceNeuronIndex, sourceNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        checkCudaErrors( cudaMemcpy ( allSynapses.destNeuronIndex, destNeuronIndex,
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

        // deep copy preSpikeQueue from host to device
        BGQUEUE_ELEMENT* pQueueBuffer; // temporary buffer to save event queue.

        // allocate device memory for the buffer.
        checkCudaErrors( cudaMalloc( ( void ** ) &pQueueBuffer, preSpikeQueue->m_nMaxEvent * sizeof( BGQUEUE_ELEMENT ) ) );

        // copy event queue data from host to the buffer.
        checkCudaErrors( cudaMemcpy ( pQueueBuffer, preSpikeQueue->m_queueEvent, preSpikeQueue->m_nMaxEvent * sizeof( BGQUEUE_ELEMENT ), cudaMemcpyHostToDevice ) );

        // copy event queue data from the buffer to the device.
        copyEventQueueDevice <<< 1, 1 >>> (allSynapses.preSpikeQueue, preSpikeQueue->m_nMaxEvent, preSpikeQueue->m_idxQueue, pQueueBuffer);

        // free device memory for the buffer.
        checkCudaErrors( cudaFree( pQueueBuffer ) );
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingSynapses::copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info ) {
        // copy everything necessary
        AllSpikingSynapsesDeviceProperties allSynapses;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        copyDeviceToHost( allSynapses, sim_info );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHost)
 *
 *  @param  allSynapsesDevice     Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                                on device memory.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapses::copyDeviceToHost( AllSpikingSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info ) {
        int num_neurons = sim_info->totalNeurons;
        BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * num_neurons;

        checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts,
                num_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuron = allSynapses.maxSynapsesPerNeuron;
        total_synapse_counts = allSynapses.total_synapse_counts;
        count_neurons = allSynapses.count_neurons;

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapses.count_neurons = 0;

        checkCudaErrors( cudaMemcpy ( sourceNeuronIndex, allSynapses.sourceNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( destNeuronIndex, allSynapses.destNeuronIndex,
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

        // deep copy preSpikeQueue from device to host.
        BGQUEUE_ELEMENT* pQueueBuffer; // temporary buffer to save event queue.
        EventQueue* pDstEventQueue;    // temporary buffer to save EventQueue object.

        // allocate device memories for buffers.
        checkCudaErrors( cudaMalloc( ( void ** ) &pQueueBuffer, preSpikeQueue->m_nMaxEvent * sizeof( BGQUEUE_ELEMENT ) ) );
        checkCudaErrors( cudaMalloc( ( void ** ) &pDstEventQueue, sizeof( EventQueue ) ) );

        // copy event queue data from device to the buffers.
        copyEventQueueDevice <<< 1, 1 >>> (allSynapses.preSpikeQueue, pQueueBuffer, pDstEventQueue);

        // copy data in the buffers to the event queue in host memory.
        checkCudaErrors( cudaMemcpy ( preSpikeQueue->m_queueEvent, pQueueBuffer, preSpikeQueue->m_nMaxEvent * sizeof( BGQUEUE_ELEMENT ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( &preSpikeQueue->m_nMaxEvent, &pDstEventQueue->m_nMaxEvent, sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( &preSpikeQueue->m_idxQueue, &pDstEventQueue->m_idxQueue, sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );

        // free device memories for buffers.
        checkCudaErrors( cudaFree( pQueueBuffer ) );
        checkCudaErrors( cudaFree( pDstEventQueue ) );
}

/*
 *  Get synapse_counts in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceSynapseCountsToHost(void* allSynapsesDevice, const SimulationInfo *sim_info)
{
        AllSpikingSynapsesDeviceProperties allSynapses;
        int neuron_count = sim_info->totalNeurons;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts, neuron_count * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        //allSynapses.count_neurons = 0;
}

/* 
 *  Get summationCoord and in_use in AllSynapses struct on device memory.
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllSpikingSynapses::copyDeviceSynapseSumIdxToHost(void* allSynapsesDevice, const SimulationInfo *sim_info)
{
        AllSpikingSynapsesDeviceProperties allSynapses;
        BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * sim_info->totalNeurons;

        checkCudaErrors( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        checkCudaErrors( cudaMemcpy ( sourceNeuronIndex, allSynapses.sourceNeuronIndex,
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
 *  Advance all the Synapses in the simulation.
 *  Update the state of all synapses for a time step.
 *
 *  @param  allSynapsesDevice      Reference to the AllSynapsesDeviceProperties struct 
 *                                 on device memory.
 *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 *  @param  sim_info               SimulationInfo class to read information from.
 */
void AllSpikingSynapses::advanceSynapses(void* allSynapsesDevice, void* allNeuronsDevice, void* synapseIndexMapDevice, const SimulationInfo *sim_info)
{
    if (total_synapse_counts == 0)
        return;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSpikingSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, (SynapseIndexMap*)synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice );

    advanceSpikingSynapsesEventQueueDevice <<< 1, 1 >>> ((AllSpikingSynapsesDeviceProperties*)allSynapsesDevice);
}
