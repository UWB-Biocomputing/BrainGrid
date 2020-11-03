/*
 *      \file GpuSInputRegular.cu
 *
 *      \author Fumitaka Kawasaki
 *
 *      \brief A class that performs stimulus input (implementation Regular) on GPU.
 */

#include "GpuSInputRegular.h"
#include <helper_cuda.h>

/*
 * constructor
 *
 * @param[in] psi          Pointer to the simulation information
 * @param[in] firingRate   Firing Rate (Hz)
 * @param[in] duration     Duration of a pulse in second
 * @param[in] interval     Interval between pulses in second
 * @param[in] weight       Synapse weight
 * @param[in] maskIndex    Input masks index
 */
GpuSInputRegular::GpuSInputRegular(SimulationInfo* psi, BGFLOAT firingRate, BGFLOAT duration, BGFLOAT interval, string const &sync, BGFLOAT weight, vector<BGFLOAT> &maskIndex) : SInputRegular(psi, firingRate, duration, interval, sync, weight, maskIndex)
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
        allocDeviceValues(psi, vtClrInfo[iCluster], m_nISIs, m_nShiftValues);
    }
}

/*
 * Terminate process.
 *
 * @param[in] psi             Pointer to the simulation information.
 * @param[in] vtClrInfo       Vector of ClusterInfo.
 */
void GpuSInputRegular::term(SimulationInfo* psi, vector<ClusterInfo *> const &vtClrInfo)
{
    if (m_fSInput) {
        // for each cluster
        for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClrInfo.size(); iCluster++) {
            checkCudaErrors( cudaSetDevice( vtClrInfo[iCluster]->deviceId ) );

            deleteDeviceValues(vtClrInfo[iCluster]);
        }
    }

    SInputRegular::term(psi, vtClrInfo);
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

    int neuron_count = pci->totalClusterNeurons;
    int synapse_count = pci->totalClusterNeurons;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // add input spikes to each synapse
    inputStimulusDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, pci->nISIs_d, pci->masks_d, pci->nShiftValues_d, pci->nStepsInCycle, m_nISI, m_nStepsCycle, m_nStepsDuration, pci->synapsesPropsDeviceSInput, pci->clusterID, iStepOffset );

    // advance synapses
    int maxSpikes = (int)((psi->epochDuration * psi->maxFiringRate));
    advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( synapse_count, pci->synapseIndexMapDeviceSInput, g_simulationStep, maxSpikes, psi->deltaT, iStepOffset, pci->synapsesDeviceSInput, NULL, NULL );

    // update summation point
    applyI2SummationMap <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, pci->pClusterSummationMap, pci->synapsesPropsDeviceSInput );

    // update cycle count
    pci->nStepsInCycle = (pci->nStepsInCycle + 1) % m_nStepsCycle;
}

/*
 * Advance input stimulus state.
 *
 * @param[in] pci             ClusterInfo class to read information from.
 * @param[in] iStep           Simulation steps to advance.
 */
void GpuSInputRegular::advanceSInputState(const ClusterInfo *pci, int iStep)
{
    // Advances synapses pre spike event queue state of the cluster iStep simulation step
    advanceSpikeQueueDevice <<< 1, 1 >>> (iStep, pci->synapsesDeviceSInput);
}

/*
 * Allocate GPU device memory and copy values
 *
 * @param[in] psi               allocDeviceValuesPointer to the simulation information.
 * @param[in] pci               Pointer to the cluster information.
 * @param[in] nISIs             Pointer to the interval counter.
 * @param[in] nShiftValues      Pointer to the shift values.
 */
void GpuSInputRegular::allocDeviceValues( SimulationInfo* psi, ClusterInfo* pci, int *nISIs, int *nShiftValues )
{
    int neuron_count = pci->totalClusterNeurons;

    // Allocate GPU device memory
    BGSIZE nISIs_d_size = neuron_count * sizeof (int);   // size of shift values
    checkCudaErrors( cudaMalloc ( ( void ** ) &(pci->nISIs_d), nISIs_d_size ) );

    BGSIZE nShiftValues_d_size = neuron_count * sizeof (int);   // size of shift values
    checkCudaErrors( cudaMalloc ( ( void ** ) &pci->nShiftValues_d, nShiftValues_d_size ) );

    // Copy values into device memory
   int beginIdx = pci->clusterNeuronsBegin;
    checkCudaErrors( cudaMemcpy ( pci->nISIs_d, &nISIs[beginIdx], nISIs_d_size, cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( pci->nShiftValues_d, &nShiftValues[beginIdx], nShiftValues_d_size, cudaMemcpyHostToDevice ) );

    // create an input synapse layer in device
    AllSynapsesProps *pSynapsesProps = dynamic_cast<AllSynapses*>(pci->synapsesSInput)->m_pSynapsesProps;
    pSynapsesProps->setupSynapsesDeviceProps( (void **)&(pci->synapsesPropsDeviceSInput), neuron_count, 1 );
    pSynapsesProps->copySynapseHostToDeviceProps( pci->synapsesPropsDeviceSInput, neuron_count, 1 );

    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    initSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( pci->synapsesDeviceSInput, neuron_count, pci->synapsesPropsDeviceSInput, pci->pClusterSummationMap, psi->width, psi->deltaT, m_weight );

    // allocate memory for curand global state
    checkCudaErrors( cudaMalloc ( &(pci->devStates_d), neuron_count * sizeof( curandState ) ) );

    // allocate memory for synapse index map and initialize it
    SynapseIndexMap synapseIndexMap;
    BGSIZE* incomingSynapseIndexMap = new BGSIZE[neuron_count];

    BGSIZE syn_i = 0;
    for (int i = 0; i < neuron_count; i++, syn_i++)
    {
        incomingSynapseIndexMap[i] = syn_i;
    }

    checkCudaErrors( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseIndexMap, neuron_count * sizeof( BGSIZE ) ) );
    checkCudaErrors( cudaMemcpy ( synapseIndexMap.incomingSynapseIndexMap, incomingSynapseIndexMap, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &(pci->synapseIndexMapDeviceSInput), sizeof( SynapseIndexMap ) ) );
    checkCudaErrors( cudaMemcpy ( pci->synapseIndexMapDeviceSInput, &synapseIndexMap, sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );

    delete[] incomingSynapseIndexMap;

    // allocate memory for masks for stimulus input and initialize it
    checkCudaErrors( cudaMalloc ( &(pci->masks_d), neuron_count * sizeof( bool ) ) );
    checkCudaErrors( cudaMemcpy ( pci->masks_d, &m_masks[beginIdx], neuron_count * sizeof( bool ), cudaMemcpyHostToDevice ) );

    // create an AllSynapses class object in device
    pci->synapsesSInput->createAllSynapsesInDevice(&(pci->synapsesDeviceSInput), pci->synapsesPropsDeviceSInput);
}

/* 
 * Dellocate GPU device memory 
 *
 * @param[in] pci               Pointer to the cluster information.
 */ 
void GpuSInputRegular::deleteDeviceValues( ClusterInfo* pci )
{   
    checkCudaErrors( cudaFree( pci->nISIs_d ) );
    checkCudaErrors( cudaFree( pci->nShiftValues_d ) );
    checkCudaErrors( cudaFree( pci->masks_d ) );

    AllSynapsesProps *pSynapsesProps = dynamic_cast<AllSynapses*>(pci->synapsesSInput)->m_pSynapsesProps;
    pSynapsesProps->cleanupSynapsesDeviceProps( pci->synapsesPropsDeviceSInput );

    // deallocate memory for synapse index map
    SynapseIndexMap synapseIndexMap;
    checkCudaErrors( cudaMemcpy ( &synapseIndexMap, pci->synapseIndexMapDeviceSInput, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaFree( synapseIndexMap.incomingSynapseIndexMap ) );
    checkCudaErrors( cudaFree( pci->synapseIndexMapDeviceSInput ) );

    // delete an AllSynapses class object in device
    pci->synapsesSInput->deleteAllSynapsesInDevice(pci->synapsesDeviceSInput);
}

// CUDA code for -----------------------------------------------------------------------
/*
 * Device code for adding input values to the summation map.
 *
 * @param[in] nISIs_d           Pointer to the interval counter.
 * @param[in] masks_d           Pointer to the input stimulus masks.
 * @param[in] nShiftValues_d    Pointer to the shift values.
 * @param[in] nISI              Spikes interval in simulation steps.
 * @param[in] nStepsInCycle     Current steps in cycle
 * @param[in] nStepsCycle       Number of steps in one cycle
 * @param[in] nStepsDuration    Number of steps in duration
 * @param[in] allSynapsesProps  Pointer to Synapse structures in device memory.
 * @param[in] clusterID         Cluster ID.
 * @param[in] iStepOffset       Offset from the current simulation step.
 */
__global__ void inputStimulusDevice( int n, int* nISIs_d, bool* masks_d, int* nShiftValues_d, int nStepsInCycle, int nISI, int nStepsCycle, int nStepsDuration, AllDSSynapsesProps* allSynapsesProps, CLUSTER_INDEX_TYPE clusterID, int iStepOffset )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;
 
    if (masks_d[idx] == false)
        return;

    BGSIZE iSyn = idx;

    int rnISIs = nISIs_d[idx];    // load the value to a register
    int rnShiftValues = nShiftValues_d[idx];	// load the value to a register

    if ( (nStepsInCycle >= rnShiftValues) && (nStepsInCycle < (rnShiftValues + nStepsDuration ) % nStepsCycle) )
    {
        if (--rnISIs <= 0)
        {
            // add a spike
            allSynapsesProps->preSpikeQueue->addAnEvent(iSyn, clusterID, iStepOffset);
            rnISIs = nISI;
        }
        else
        {
            rnISIs = 0;
        }
    }
    nISIs_d[idx] = rnISIs;
}
