/*
 * AllSpikingSynapses.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "Book.h"

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

        HANDLE_ERROR( cudaMalloc( allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );
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

        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.sourceNeuronIndex, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.destNeuronIndex, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.W, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.type, max_total_synapses * sizeof( synapseType ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.psr, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.in_use, max_total_synapses * sizeof( bool ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.synapse_counts, num_neurons * sizeof( BGSIZE ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.decay, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.tau, max_total_synapses * sizeof( BGFLOAT ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.total_delay, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayQueue, max_total_synapses * sizeof( uint32_t ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.delayIdx, max_total_synapses * sizeof( int ) ) );
        HANDLE_ERROR( cudaMalloc( ( void ** ) &allSynapses.ldelayQueue, max_total_synapses * sizeof( int ) ) );
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

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

        deleteDeviceStruct( allSynapses );

        HANDLE_ERROR( cudaFree( allSynapsesDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteSynapseDeviceStruct)
 *
 *  @param  allSynapsesDevice  Reference to the AllSpikingSynapsesDeviceProperties struct 
 *                             on device memory.
 */
void AllSpikingSynapses::deleteDeviceStruct( AllSpikingSynapsesDeviceProperties& allSynapses ) {
        HANDLE_ERROR( cudaFree( allSynapses.sourceNeuronIndex ) );
        HANDLE_ERROR( cudaFree( allSynapses.destNeuronIndex ) );
        HANDLE_ERROR( cudaFree( allSynapses.W ) );
        HANDLE_ERROR( cudaFree( allSynapses.type ) );
        HANDLE_ERROR( cudaFree( allSynapses.psr ) );
        HANDLE_ERROR( cudaFree( allSynapses.in_use ) );
        HANDLE_ERROR( cudaFree( allSynapses.synapse_counts ) );
        HANDLE_ERROR( cudaFree( allSynapses.decay ) );
        HANDLE_ERROR( cudaFree( allSynapses.tau ) );
        HANDLE_ERROR( cudaFree( allSynapses.total_delay ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayQueue ) );
        HANDLE_ERROR( cudaFree( allSynapses.delayIdx ) );
        HANDLE_ERROR( cudaFree( allSynapses.ldelayQueue ) );

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

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

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
        HANDLE_ERROR( cudaMemcpy ( allSynapsesDevice, &allSynapses, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyHostToDevice ) );

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapses.count_neurons = 0;

        HANDLE_ERROR( cudaMemcpy ( allSynapses.sourceNeuronIndex, sourceNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.destNeuronIndex, destNeuronIndex,
                max_total_synapses * sizeof( int ),  cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.W, W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.type, type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.psr, psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.in_use, in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.synapse_counts, synapse_counts,
                        num_neurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.decay, decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.tau, tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.total_delay, total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayQueue, delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.delayIdx, delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses.ldelayQueue, ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyHostToDevice ) );
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

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );

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

        HANDLE_ERROR( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts,
                num_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
        maxSynapsesPerNeuron = allSynapses.maxSynapsesPerNeuron;
        total_synapse_counts = allSynapses.total_synapse_counts;
        count_neurons = allSynapses.count_neurons;

        // Set count_neurons to 0 to avoid illegal memory deallocation 
        // at AllSpikingSynapses deconstructor.
        allSynapses.count_neurons = 0;

        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndex, allSynapses.sourceNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( destNeuronIndex, allSynapses.destNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( W, allSynapses.W,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( type, allSynapses.type,
                max_total_synapses * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( psr, allSynapses.psr,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( in_use, allSynapses.in_use,
                max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( decay, allSynapses.decay,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( tau, allSynapses.tau,
                max_total_synapses * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( total_delay, allSynapses.total_delay,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayQueue, allSynapses.delayQueue,
                max_total_synapses * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( delayIdx, allSynapses.delayIdx,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( ldelayQueue, allSynapses.ldelayQueue,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
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

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( synapse_counts, allSynapses.synapse_counts, neuron_count * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );

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

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSpikingSynapsesDeviceProperties ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( sourceNeuronIndex, allSynapses.sourceNeuronIndex,
                max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( in_use, allSynapses.in_use,
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

    HANDLE_ERROR( cudaMemcpyToSymbol(classSynapses_d, &classSynapses_h, sizeof(enumClassSynapses)) );
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
}
