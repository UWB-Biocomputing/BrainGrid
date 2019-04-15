/*
 *      \file GpuSInputRegular.cu
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Regular) on GPU.
 */

#include "GpuSInputRegular.h"
#include <helper_cuda.h>

// Forward Delaration
void allocDeviceValues( ClusterInfo* pci, BGFLOAT* initValues, int *nShiftValues );
void deleteDeviceValues( ClusterInfo* pci );

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
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void GpuSInputRegular::init(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    SInputRegular::init(psi, vtClrInfo);

    if (m_fSInput == false)
        return;

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++) {
        checkCudaErrors( cudaSetDevice( vtClrInfo[iCluster]->deviceId ) );

        // allocate GPU device memory and copy values
        allocDeviceValues(vtClrInfo[iCluster], m_values, m_nShiftValues);
    }

    delete[] m_values;
    delete[] m_nShiftValues;
}

/*
 * Terminate process.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void GpuSInputRegular::term(SimulationInfo* psi, vector<ClusterInfo *> &vtClrInfo)
{
    if (m_fSInput) {
        // for each cluster
        for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++) {
            checkCudaErrors( cudaSetDevice( vtClrInfo[iCluster]->deviceId ) );

            deleteDeviceValues(vtClrInfo[iCluster]);
        }
    }
}

/*
 * Process input stimulus for each time step on GPU.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] pci             ClusterInfo class to read information from.
 * @param[in] iStepOffset     Offset from the current simulation step.
 */
void GpuSInputRegular::inputStimulus(const SimulationInfo* psi, ClusterInfo *pci, int iStepOffset)
{
    if (m_fSInput == false)
        return;

    // for each cluster
    checkCudaErrors( cudaSetDevice( pci->deviceId ) );

    // add input to each summation point
    inputStimulusDevice <<< psi->neuronBlocksPerGrid, psi->threadsPerBlock >>> (pci->totalClusterNeurons, pci->pClusterSummationMap, pci->initValues_d, pci->nShiftValues_d, pci->nStepsInCycle, m_nStepsCycle, m_nStepsDuration );

    // update cycle count
    pci->nStepsInCycle = (pci->nStepsInCycle + 1) % m_nStepsCycle;  
}

/*
 * Allocate GPU device memory and copy values
 *
 * @param[in] pci               Pointer to the cluster information.
 * @param[in] initValues        Pointer to the initial values.
 * @param[in] nShiftValues      Pointer to the shift values.
 */
void allocDeviceValues( ClusterInfo* pci, BGFLOAT* initValues, int *nShiftValues )
{
    int neuron_count = pci->totalClusterNeurons;
    BGSIZE initValues_d_size = neuron_count * sizeof (BGFLOAT);   // size of initial values
    BGSIZE nShiftValues_d_size = neuron_count * sizeof (int);   // size of shift values

    // Allocate GPU device memory
    checkCudaErrors( cudaMalloc ( ( void ** ) &pci->initValues_d, initValues_d_size ) );
    checkCudaErrors( cudaMalloc ( ( void ** ) &pci->nShiftValues_d, nShiftValues_d_size ) );

    // Copy values into device memory
    checkCudaErrors( cudaMemcpy ( pci->initValues_d, &initValues[pci->clusterNeuronsBegin], initValues_d_size, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( pci->nShiftValues_d, &nShiftValues[pci->clusterNeuronsBegin], nShiftValues_d_size, cudaMemcpyHostToDevice ) );
}

/* 
 * Dellocate GPU device memory 
 *
 * @param[in] pci               Pointer to the cluster information.
 */ 
void deleteDeviceValues( ClusterInfo* pci )
{   
    checkCudaErrors( cudaFree( pci->initValues_d ) );
    checkCudaErrors( cudaFree( pci->nShiftValues_d ) );
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
