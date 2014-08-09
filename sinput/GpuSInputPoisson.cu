/**
 *      \file GpuSInputPoisson.cu
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson) on GPU.
 */

#include "GpuSInputPoisson.h"
#include "curand_kernel.h"
#include "DelayIdx.h"
#include "DynamicSpikingSynapse_struct.h"
#include "book.h"

// Forward Delaration
void allocDeviceValues( SimulationInfo* psi, int *nISIs );
void deleteDeviceValues( );

//! Device function that processes input stimulus for each time step.
__global__ void inputStimulusDevice( int n, int* nISIs_d, BGFLOAT deltaT, int delay, BGFLOAT lambda, curandState* devStates_d );
__global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d );
__global__ void setupSeeds( int n, curandState* devStates_d, unsigned long seed );

extern __global__ void advanceSynapsesDevice( int n, DynamicSpikingSynapse_struct lsynapse_st_d, uint64_t simulationStep, uint32_t bmask );
extern void allocSynapseStruct_d( int count, DynamicSpikingSynapse_struct& lsynapse_st );
extern void deleteSynapseStruct_d( DynamicSpikingSynapse_struct& lsynapse_st_d );
extern void copySynapseHostToDevice( DynamicSpikingSynapse_struct& synapse_h, DynamicSpikingSynapse_struct& lsynapse_st_d, int count );

//! Pointer to device interval counter.
int * nISIs_d = NULL;

//! Delayed queue index - global to all synapses.
extern DelayIdx delayIdx;

//! Synapse structures in device constant memory.
__constant__ DynamicSpikingSynapse_struct isynapse_st_d[1];

//! Memory to save global state for curand.
curandState* devStates_d;

/**
 * constructor
 */
GpuSInputPoisson::GpuSInputPoisson() : SInputPoisson()
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
 * @param[in] psi       Pointer to the simulation information.
 * @param[in] parms     Pointer to xml parms element
 */
void GpuSInputPoisson::init(SimulationInfo* psi, TiXmlElement* parms)
{
    SInputPoisson::init(psi, parms);

    if (fSInput == false)
        return;

    // allocate GPU device memory and copy values
    allocDeviceValues(psi, nISIs);

    // CUDA parameters
    int neuron_count = psi->cNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // setup seeds
    setupSeeds <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, devStates_d, time(NULL) );

    // delete host memory
    delete[] nISIs;
    synapseList.clear();
}

/**
 * Terminate process.
 */
void GpuSInputPoisson::term()
{
    SInputPoisson::term();

    if (fSInput)
        deleteDeviceValues( );
}

/**
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 * @param[in] psi                Pointer to the simulation information.
 * @param[in] summationPoint_d   summationPoint
 */
void GpuSInputPoisson::inputStimulus(SimulationInfo* psi, BGFLOAT* summationPoint_d)
{
    if (fSInput == false)
        return;

    int neuron_count = psi->cNeurons;
    int synapse_count = psi->cNeurons;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // add input spikes to each synapse
    inputStimulusDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, nISIs_d, psi->deltaT, delayIdx.getIndex(), lambda, devStates_d );

    // advance synapses
    uint32_t bmask = delayIdx.getBitmask(  );
    DynamicSpikingSynapse_struct lsynapse_st;
    HANDLE_ERROR( cudaMemcpyFromSymbol ( &lsynapse_st, isynapse_st_d, sizeof( DynamicSpikingSynapse_struct ) ) );
    advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( synapse_count, lsynapse_st, g_simulationStep, bmask );

    // update summation point
    applyI2SummationMap <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, summationPoint_d );
}

/**
 * Allocate GPU device memory and copy values
 * @param[in] psi        Pointer to the simulation information.
 * @param[in] nISIs      Pointer to the interval counter.
 */
void GpuSInputPoisson::allocDeviceValues( SimulationInfo* psi, int *nISIs )
{
    int neuron_count = psi->cNeurons;
    size_t nISIs_d_size = neuron_count * sizeof (int);   // size of shift values

    // Allocate GPU device memory
    HANDLE_ERROR( cudaMalloc ( ( void ** ) &nISIs_d, nISIs_d_size ) );

    // Copy values into device memory
    HANDLE_ERROR( cudaMemcpy ( nISIs_d, nISIs, nISIs_d_size, cudaMemcpyHostToDevice ) );

    // allocate device memory for synapses
    int synapse_count = psi->cNeurons;
    DynamicSpikingSynapse_struct isynapse_st;
    allocSynapseStruct_d( synapse_count, isynapse_st ); 
    HANDLE_ERROR( cudaMemcpyToSymbol ( isynapse_st_d, &isynapse_st, sizeof( DynamicSpikingSynapse_struct ) ) );

    // copy synapse into arrays
    DynamicSpikingSynapse_struct synapse_st;
    allocSynapseStruct(synapse_st, neuron_count);
    for (int i = 0; i < synapse_count; i++)
    {
        copySynapseToStruct(synapseList[i], synapse_st, i);
    }

    // copy synapse data into device memory
    DynamicSpikingSynapse_struct lsynapse_st_d;
    HANDLE_ERROR( cudaMemcpyFromSymbol ( &lsynapse_st_d, isynapse_st_d, sizeof( DynamicSpikingSynapse_struct ) ) );
    copySynapseHostToDevice( synapse_st, lsynapse_st_d, synapse_count );

    // delete the arrays
    deleteSynapseStruct(synapse_st);

    // allocate memory for curand global state
    HANDLE_ERROR( cudaMalloc ( &devStates_d, neuron_count*sizeof( curandState ) ) );
}

/**
 * Dellocate GPU device memory
 */
void GpuSInputPoisson::deleteDeviceValues(  )
{
    HANDLE_ERROR( cudaFree( nISIs_d ) );
    DynamicSpikingSynapse_struct synapse_st;
    HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse_st, isynapse_st_d, sizeof( DynamicSpikingSynapse_struct ) ) );
    deleteSynapseStruct_d( synapse_st  );
    HANDLE_ERROR( cudaFree( devStates_d ) );
}

// CUDA code for -----------------------------------------------------------------------
/**
 * Device code for adding input values to the summation map.
 * @param[in] nISIs_d           Pointer to the interval counter.
 * @param[in] deltaT            Time step of the simulation in second.
 * @param[in] lambda            Iinverse firing rate.
 * @param[in] devStates_d        Curand global state
 */
__global__ void inputStimulusDevice( int n, int* nISIs_d, BGFLOAT deltaT, int delay, BGFLOAT lambda, curandState* devStates_d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    int rnISIs = nISIs_d[idx];    // load the value to a register
    if (--rnISIs <= 0)
    {
        // add a spike
        int idx0 = delay + isynapse_st_d[0].total_delay[idx];
        if ( idx0 >= LENGTH_OF_DELAYQUEUE )
            idx0 -= LENGTH_OF_DELAYQUEUE;
        isynapse_st_d[0].delayQueue[idx] |= (0x1 << idx0);

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
 */
__global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
            return;

    summationPoint_d[idx] += isynapse_st_d[0].psr[idx];
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
