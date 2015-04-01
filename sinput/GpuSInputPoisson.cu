/**
 *      \file GpuSInputPoisson.cu
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson) on GPU.
 */

#include "GpuSInputPoisson.h"
#include "curand_kernel.h"
#include "Book.h"

//! Device function that processes input stimulus for each time step.
__global__ void initSynapsesDevice( int n, AllDSSynapses* allSynapsesDevice, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight );
__global__ void inputStimulusDevice( int n, int* nISIs_d, bool* masks_d, BGFLOAT deltaT, BGFLOAT lambda, curandState* devStates_d, AllDSSynapses* allSynapsesDevice );
__global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapses* allSynapsesDevice );
__global__ void setupSeeds( int n, curandState* devStates_d, unsigned long seed );

extern __global__ void advanceSynapsesDevice ( int total_synapse_counts, GPUSpikingModel::SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllDSSynapses* allSynapsesDevice );
extern __device__ void createSynapse(AllDSSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type);

//! Memory to save global state for curand.
curandState* devStates_d;

/**
 * constructor
 * @param[in] psi       Pointer to the simulation information
 */
GpuSInputPoisson::GpuSInputPoisson(SimulationInfo* psi, TiXmlElement* parms) : SInputPoisson(psi, parms)
{
}

/**
 * destructor
 */
GpuSInputPoisson::~GpuSInputPoisson()
{
}

/**
 * Initialize data.
 * @param[in] model     Pointer to the Neural Network Model object.
 * @param[in] neurons   The Neuron list to search from.
 * @param[in] psi       Pointer to the simulation information.
 */
void GpuSInputPoisson::init(IModel* model, AllNeurons &neurons, SimulationInfo* psi)
{
    SInputPoisson::init(model, neurons, psi);

    if (fSInput == false)
        return;

    // allocate GPU device memory and copy values
    allocDeviceValues(model, psi, nISIs);

    // CUDA parameters
    int neuron_count = psi->totalNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // setup seeds
    setupSeeds <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, devStates_d, time(NULL) );
}

/**
 * Terminate process.
 * @param[in] model              Pointer to the Neural Network Model object.
 * @param[in] psi                Pointer to the simulation information.
 */
void GpuSInputPoisson::term(IModel* model, SimulationInfo* psi)
{
    SInputPoisson::term(model, psi);

    if (fSInput)
        deleteDeviceValues(model, psi);
}

/**
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 * @param[in] model              Pointer to the Neural Network Model object.
 * @param[in] psi                Pointer to the simulation information.
 * @param[in] summationPoint_d   summationPoint
 */
void GpuSInputPoisson::inputStimulus(IModel* model, SimulationInfo* psi, BGFLOAT* summationPoint_d)
{
    if (fSInput == false)
        return;

    int neuron_count = psi->totalNeurons;
    int synapse_count = psi->totalNeurons;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // add input spikes to each synapse
    inputStimulusDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, nISIs_d, masks_d, psi->deltaT, lambda, devStates_d, allSynapsesDevice );

    // advance synapses
    advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( synapse_count, synapseIndexMapDevice, g_simulationStep, psi->deltaT, allSynapsesDevice );

    // update summation point
    applyI2SummationMap <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, summationPoint_d, allSynapsesDevice );
}

/**
 * Allocate GPU device memory and copy values
 * @param[in] model      Pointer to the Neural Network Model object.
 * @param[in] psi        Pointer to the simulation information.
 * @param[in] nISIs      Pointer to the interval counter.
 */
void GpuSInputPoisson::allocDeviceValues(IModel* model, SimulationInfo* psi, int *nISIs )
{
    int neuron_count = psi->totalNeurons;
    size_t nISIs_d_size = neuron_count * sizeof (int);   // size of shift values

    // Allocate GPU device memory
    HANDLE_ERROR( cudaMalloc ( ( void ** ) &nISIs_d, nISIs_d_size ) );

    // Copy values into device memory
    HANDLE_ERROR( cudaMemcpy ( nISIs_d, nISIs, nISIs_d_size, cudaMemcpyHostToDevice ) );

    // create an input synapse layer
    synapses->allocSynapseDeviceStruct( (void **)&allSynapsesDevice, neuron_count, 1 ); 
    synapses->copySynapseHostToDevice( allSynapsesDevice, neuron_count, 1 );

    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    initSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, allSynapsesDevice, psi->pSummationMap, psi->width, psi->deltaT, weight );

    // allocate memory for curand global state
    HANDLE_ERROR( cudaMalloc ( &devStates_d, neuron_count * sizeof( curandState ) ) );

    // allocate memory for synapse index map and initialize it
    GPUSpikingModel::SynapseIndexMap synapseIndexMap;
    uint32_t* activeSynapseIndex = new uint32_t[neuron_count];

    uint32_t syn_i = 0;
    for (int i = 0; i < neuron_count; i++, syn_i++)
    {
        activeSynapseIndex[i] = syn_i;
    }
    HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.activeSynapseIndex, neuron_count * sizeof( uint32_t ) ) );
    HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.activeSynapseIndex, activeSynapseIndex, neuron_count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) ); 
    HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMapDevice, sizeof( GPUSpikingModel::SynapseIndexMap ) ) );
    HANDLE_ERROR( cudaMemcpy ( synapseIndexMapDevice, &synapseIndexMap, sizeof( GPUSpikingModel::SynapseIndexMap ), cudaMemcpyHostToDevice ) );

    delete[] activeSynapseIndex;

    // allocate memory for masks for stimulus input and initialize it
    HANDLE_ERROR( cudaMalloc ( &masks_d, neuron_count * sizeof( bool ) ) );
    HANDLE_ERROR( cudaMemcpy ( masks_d, masks, neuron_count * sizeof( bool ), cudaMemcpyHostToDevice ) ); 
}

/**
 * Dellocate GPU device memory
 * @param[in] model      Pointer to the Neural Network Model object.
 * @param[in] psi        Pointer to the simulation information.
 */
void GpuSInputPoisson::deleteDeviceValues(IModel* model, SimulationInfo* psi )
{
    int neuron_count = psi->totalNeurons;

    HANDLE_ERROR( cudaFree( nISIs_d ) );
    HANDLE_ERROR( cudaFree( devStates_d ) );
    HANDLE_ERROR( cudaFree( masks_d ) );

    synapses->deleteSynapseDeviceStruct( allSynapsesDevice, neuron_count, 1 );

    // deallocate memory for synapse index map
    GPUSpikingModel::SynapseIndexMap synapseIndexMap;
    HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, sizeof( GPUSpikingModel::SynapseIndexMap ), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaFree( synapseIndexMap.activeSynapseIndex ) );
    HANDLE_ERROR( cudaFree( synapseIndexMapDevice ) );
}

// CUDA code for -----------------------------------------------------------------------

/** 
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 * @param allSynapsesDevice      Pointer to the Synapse structures in device memory.
 * @param pSummationMap          Pointer to the summation point.
 * @param width                  Width of neuron map (assumes square).
 * @param deltaT                 The time step size.
 * @param weight                 Synapse weight.
 */
__global__ void initSynapsesDevice( int n, AllDSSynapses* allSynapsesDevice, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    // create a synapse
    int neuron_index = idx;
    int dest_x = neuron_index % width;;
    int dest_y = neuron_index / width;;
    BGFLOAT* sum_point = &( pSummationMap[neuron_index] );
    synapseType type = allSynapsesDevice->type[neuron_index];
    createSynapse(allSynapsesDevice, neuron_index, 0, 0, 0, dest_x, dest_y, sum_point, deltaT, type );
    allSynapsesDevice->W[neuron_index] = weight * SYNAPSE_STRENGTH_ADJUSTMENT;
}

/**
 * Device code for adding input values to the summation map.
 * @param[in] nISIs_d           Pointer to the interval counter.
 * @param[in] masks_d           Pointer to the input stimulus masks.
 * @param[in] deltaT            Time step of the simulation in second.
 * @param[in] lambda            Iinverse firing rate.
 * @param[in] devStates_d        Curand global state
 * @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
 */
__global__ void inputStimulusDevice( int n, int* nISIs_d, bool* masks_d, BGFLOAT deltaT, BGFLOAT lambda, curandState* devStates_d, AllDSSynapses* allSynapsesDevice )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    if (masks_d[idx] == false)
        return;

    uint32_t iSyn = idx;

    int rnISIs = nISIs_d[idx];    // load the value to a register
    if (--rnISIs <= 0)
    {
        // add a spike
        uint32_t &delay_queue = allSynapsesDevice->delayQueue[iSyn];
        int delayIdx = allSynapsesDevice->delayIdx[iSyn];
        int ldelayQueue = allSynapsesDevice->ldelayQueue[iSyn];
        int total_delay = allSynapsesDevice->total_delay[iSyn];

        // Add to spike queue

        // calculate index where to insert the spike into delayQueue
        int idx = delayIdx +  total_delay;
        if ( idx >= ldelayQueue ) {
            idx -= ldelayQueue;
        }

        // set a spike
        //assert( !(delay_queue[0] & (0x1 << idx)) );
        delay_queue |= (0x1 << idx);

        // update interval counter (exponectially distribution ISIs, Poisson)
        curandState localState = devStates_d[idx];

        BGFLOAT isi = -lambda * log(curand_uniform( &localState ));
        // delete isi within refractoriness
        while (curand_uniform( &localState ) <= exp(-(isi*isi)/32))
            isi = -lambda * log(curand_uniform( &localState ));
        // convert isi from msec to steps
        rnISIs = static_cast<int>( (isi / 1000) / deltaT + 0.5 );
        devStates_d[idx] = localState;
    }
    nISIs_d[idx] = rnISIs;
}

// CUDA code for update summation point -----------------------------------------------------
/**
 * @param[in] n                  Number of neurons.
 * @param[in] summationPoint_d   SummationPoint
 * @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
 */
__global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapses* allSynapsesDevice ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
            return;

    summationPoint_d[idx] += allSynapsesDevice->psr[idx];
}

// CUDA code for setup curand seed -----------------------------------------------------
/**
 * @param[in] n                  Number of neurons.
 * @param[in] devStates_d        Curand global state
 * @param[in] seed               Seed
 */
__global__ void setupSeeds( int n, curandState* devStates_d, unsigned long seed )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
            return;

    curand_init( seed, idx, 0, &devStates_d[idx] );
} 
