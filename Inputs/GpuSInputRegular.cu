/*
 *      \file GpuSInputRegular.cu
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Regular) on GPU.
 */

#include "GpuSInputRegular.h"
#include "Book.h"

// Forward Delaration
void allocDeviceValues( SimulationInfo* psi, BGFLOAT* initValues, int *nShiftValues );
void deleteDeviceValues( );

//! Pointer to device input values.
BGFLOAT* initValues_d = NULL;
int * nShiftValues_d = NULL;

/*
 * constructor
 *
 * @param[in] psi       Pointer to the simulation information
 * @param[in] parms     TiXmlElement to examine.
 */
GpuSInputRegular::GpuSInputRegular(SimulationInfo* psi, TiXmlElement* parms) : SInputRegular(psi, parms)
{
}

/*
 * destructor
 */
GpuSInputRegular::~GpuSInputRegular()
{
}

/*
 * Initialize data.
 *
 * @param[in] psi       Pointer to the simulation information.
 */
void GpuSInputRegular::init(SimulationInfo* psi)
{
    SInputRegular::init(psi);

    if (fSInput == false)
        return;

    // allocate GPU device memory and copy values
    allocDeviceValues(psi, values, nShiftValues);

    delete[] values;
    delete[] nShiftValues;
}

/*
 * Terminate process.
 *
 * @param[in] psi       Pointer to the simulation information.
 */
void GpuSInputRegular::term(SimulationInfo* psi)
{
    if (fSInput)
        deleteDeviceValues( );
}

/*
 * Process input stimulus for each time step on GPU.
 *
 * @param[in] psi               Pointer to the simulation information.
 */
void GpuSInputRegular::inputStimulus(SimulationInfo* psi)
{
    if (fSInput == false)
        return;

    int neuron_count = psi->totalNeurons;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid; 

    // add input to each summation point
    blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
    inputStimulusDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, psi->pSummationMap, initValues_d, nShiftValues_d, nStepsInCycle, nStepsCycle, nStepsDuration );
    // update cycle count
    nStepsInCycle = (nStepsInCycle + 1) % nStepsCycle;
}

/*
 * Allocate GPU device memory and copy values
 *
 * @param[in] psi               Pointer to the simulation information.
 * @param[in] initValues        Pointer to the initial values.
 * @param[in] nShiftValues      Pointer to the shift values.
 */
void allocDeviceValues( SimulationInfo* psi, BGFLOAT* initValues, int *nShiftValues )
{
    int neuron_count = psi->totalNeurons;
    BGSIZE initValues_d_size = neuron_count * sizeof (BGFLOAT);   // size of initial values
    BGSIZE nShiftValues_d_size = neuron_count * sizeof (int);   // size of shift values

    // Allocate GPU device memory
    HANDLE_ERROR( cudaMalloc ( ( void ** ) &initValues_d, initValues_d_size ) );
    HANDLE_ERROR( cudaMalloc ( ( void ** ) &nShiftValues_d, nShiftValues_d_size ) );

    // Copy values into device memory
    HANDLE_ERROR( cudaMemcpy ( initValues_d, initValues, initValues_d_size, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy ( nShiftValues_d, nShiftValues, nShiftValues_d_size, cudaMemcpyHostToDevice ) );
}

/* 
 * Dellocate GPU device memory 
 */ 
void deleteDeviceValues(  )
{   
    HANDLE_ERROR( cudaFree( initValues_d ) );
    HANDLE_ERROR( cudaFree( nShiftValues_d ) );
}

// CUDA code for -----------------------------------------------------------------------
/*
 * Device code for adding input values to the summation map.
 *
 * @param[in] summationPoint_d  Pointer to the summation map.
 * @param[in] initValues_d      Pointer to the input values.
 * @param[in] nShiftValues_d    Pointer to the shift values.
 * @param[in] nStepsInCycle     Current steps in cycle
 * @param[in] nStepsCycle       Number of steps in one cycle
 * @param[in] nStepsDuration    Number of steps in duration
 */
__global__ void inputStimulusDevice( int n, BGFLOAT* summationPoint_d, BGFLOAT* initValues_d, int* nShiftValues_d, int nStepsInCycle, int nStepsCycle, int nStepsDuration )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;
 
    int rnShiftValues = nShiftValues_d[idx];	// load the value to a register
    if ( (nStepsInCycle >= rnShiftValues) && (nStepsInCycle < (rnShiftValues + nStepsDuration ) % nStepsCycle) )
    {
        BGFLOAT rinitValue = initValues_d[idx];
        if (rinitValue != 0.0)
            summationPoint_d[idx] += rinitValue;
    }
}
