/*
 *      \file GpuSInputPoisson.cu
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Poisson) on GPU.
 */

#include "curand_kernel.h"
#include "GpuSInputPoisson.h"
#include "Book.h"

//! Memory to save global state for curand.
curandState* devStates_d;

/*
 * constructor
 *
 * @param[in] psi       Pointer to the simulation information
 * @param[in] parms     TiXmlElement to examine.
 */
GpuSInputPoisson::GpuSInputPoisson(SimulationInfo* psi, TiXmlElement* parms) : SInputPoisson(psi, parms)
{
}

/*
 * destructor
 */
GpuSInputPoisson::~GpuSInputPoisson()
{
}

/*
 * Initialize data.
 *
 * @param[in] psi       Pointer to the simulation information.
 */
void GpuSInputPoisson::init(SimulationInfo* psi)
{
    SInputPoisson::init(psi);

    if (fSInput == false)
        return;

    // allocate GPU device memory and copy values
    allocDeviceValues(psi->model, psi, nISIs);

    // CUDA parameters
    int neuron_count = psi->totalNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // setup seeds
    setupSeeds <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, devStates_d, time(NULL) );
}

/*
 * Terminate process.
 *
 * @param[in] psi                Pointer to the simulation information.
 */
void GpuSInputPoisson::term(SimulationInfo* psi)
{
    SInputPoisson::term(psi);

    if (fSInput)
        deleteDeviceValues(psi->model, psi);
}

/*
 * Process input stimulus for each time step.
 * Apply inputs on summationPoint.
 *
 * @param[in] psi                Pointer to the simulation information.
 */
void GpuSInputPoisson::inputStimulus(SimulationInfo* psi)
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
    advanceSpikingSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( synapse_count, synapseIndexMapDevice, g_simulationStep, psi->deltaT, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice );

    // update summation point
    applyI2SummationMap <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, psi->pSummationMap, allSynapsesDevice );
}

/*
 * Allocate GPU device memory and copy values
 *
 * @param[in] model      Pointer to the Neural Network Model object.
 * @param[in] psi        Pointer to the simulation information.
 * @param[in] nISIs      Pointer to the interval counter.
 */
void GpuSInputPoisson::allocDeviceValues(IModel* model, SimulationInfo* psi, int *nISIs )
{
    int neuron_count = psi->totalNeurons;
    BGSIZE nISIs_d_size = neuron_count * sizeof (int);   // size of shift values

    // Allocate GPU device memory
    HANDLE_ERROR( cudaMalloc ( ( void ** ) &nISIs_d, nISIs_d_size ) );

    // Copy values into device memory
    HANDLE_ERROR( cudaMemcpy ( nISIs_d, nISIs, nISIs_d_size, cudaMemcpyHostToDevice ) );

    // create an input synapse layer
    m_synapses->allocSynapseDeviceStruct( (void **)&allSynapsesDevice, neuron_count, 1 ); 
    m_synapses->copySynapseHostToDevice( allSynapsesDevice, neuron_count, 1 );

    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    initSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, allSynapsesDevice, psi->pSummationMap, psi->width, psi->deltaT, weight );

    // allocate memory for curand global state
    HANDLE_ERROR( cudaMalloc ( &devStates_d, neuron_count * sizeof( curandState ) ) );

    // allocate memory for synapse index map and initialize it
    SynapseIndexMap synapseIndexMap;
    BGSIZE* incomingSynapseIndexMap = new BGSIZE[neuron_count];

    BGSIZE syn_i = 0;
    for (int i = 0; i < neuron_count; i++, syn_i++)
    {
        incomingSynapseIndexMap[i] = syn_i;
    }
    HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseIndexMap, neuron_count * sizeof( BGSIZE ) ) );
    HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.incomingSynapseIndexMap, incomingSynapseIndexMap, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) ); 
    HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMapDevice, sizeof( SynapseIndexMap ) ) );
    HANDLE_ERROR( cudaMemcpy ( synapseIndexMapDevice, &synapseIndexMap, sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );

    delete[] incomingSynapseIndexMap;

    // allocate memory for masks for stimulus input and initialize it
    HANDLE_ERROR( cudaMalloc ( &masks_d, neuron_count * sizeof( bool ) ) );
    HANDLE_ERROR( cudaMemcpy ( masks_d, masks, neuron_count * sizeof( bool ), cudaMemcpyHostToDevice ) ); 
}

/*
 * Dellocate GPU device memory
 *
 * @param[in] model      Pointer to the Neural Network Model object.
 * @param[in] psi        Pointer to the simulation information.
 */
void GpuSInputPoisson::deleteDeviceValues(IModel* model, SimulationInfo* psi )
{
    HANDLE_ERROR( cudaFree( nISIs_d ) );
    HANDLE_ERROR( cudaFree( devStates_d ) );
    HANDLE_ERROR( cudaFree( masks_d ) );

    m_synapses->deleteSynapseDeviceStruct( allSynapsesDevice );

    // deallocate memory for synapse index map
    SynapseIndexMap synapseIndexMap;
    HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaFree( synapseIndexMap.incomingSynapseIndexMap ) );
    HANDLE_ERROR( cudaFree( synapseIndexMapDevice ) );
}

// CUDA code for -----------------------------------------------------------------------

/*
 * Device code for adding input values to the summation map.
 *
 * @param[in] nISIs_d            Pointer to the interval counter.
 * @param[in] masks_d            Pointer to the input stimulus masks.
 * @param[in] deltaT             Time step of the simulation in second.
 * @param[in] lambda             Iinverse firing rate.
 * @param[in] devStates_d        Curand global state
 * @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
 */
__global__ void inputStimulusDevice( int n, int* nISIs_d, bool* masks_d, BGFLOAT deltaT, BGFLOAT lambda, curandState* devStates_d, AllDSSynapsesDeviceProperties* allSynapsesDevice )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    if (masks_d[idx] == false)
        return;

    BGSIZE iSyn = idx;

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

/*
 * CUDA code for update summation point
 *
 * @param[in] n                  Number of neurons.
 * @param[in] summationPoint_d   SummationPoint
 * @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
 */
__global__ void applyI2SummationMap( int n, BGFLOAT* summationPoint_d, AllDSSynapsesDeviceProperties* allSynapsesDevice ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
            return;

    summationPoint_d[idx] += allSynapsesDevice->psr[idx];
}

/*
 * CUDA code for setup curand seed
 *
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
