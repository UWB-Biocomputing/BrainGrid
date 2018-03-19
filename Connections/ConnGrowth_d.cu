#include "ConnGrowth.h"
#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "GPUSpikingCluster.h"
#include <helper_cuda.h>

/*
 *  Calculates firing rates, neuron radii change and assign new values.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  vtClr       Vector of Cluster class objects.
 *  @param  vtClrInfo   Vector of ClusterInfo.
 */
void ConnGrowth::updateConns(const SimulationInfo *sim_info, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    // Calculate growth cycle firing rate for previous period
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        // Set device ID
        checkCudaErrors( cudaSetDevice( vtClrInfo[iCluster]->deviceId ) );

        AllSpikingNeurons *neurons = dynamic_cast<AllSpikingNeurons*>(vtClr[iCluster]->m_neurons);
        int neuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin;
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;

        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++) {
            // Calculate firing rate
            assert(neurons->spikeCount[iNeuron] < max_spikes);
            (*rates)[neuronLayoutIndex] = neurons->spikeCount[iNeuron] / sim_info->epochDuration;
        }
    }

    // compute neuron radii change and assign new values
    (*outgrowth) = 1.0 - 2.0 / (1.0 + exp((m_growth.epsilon - *rates / m_growth.maxRate) / m_growth.beta));
    (*deltaR) = sim_info->epochDuration * m_growth.rho * *outgrowth;
    (*radii) += (*deltaR);
}

/*
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *
 *  @param  sim_info    SimulationInfo to refer from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  vtClr       Vector of Cluster class objects.
 *  @param  vtClrInfo   Vector of ClusterInfo.
 */
void ConnGrowth::updateSynapsesWeights(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).
    (*W) = (*area);

    BGFLOAT deltaT = sim_info->deltaT;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid;

    // allocate host memory for weight data
    // W_d_size may exceed 4GB, so the type should be 64bits integer
    uint64_t W_d_size = sim_info->totalNeurons * sim_info->totalNeurons * sizeof (BGFLOAT);
    BGFLOAT* W_h = new BGFLOAT[W_d_size];

    // and initialize it
    for ( int i = 0 ; i < sim_info->totalNeurons; i++ )
        for ( int j = 0; j < sim_info->totalNeurons; j++ )
            W_h[i * sim_info->totalNeurons + j] = (*W)(i, j);

    // destination neurons of each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        // Set device ID
        checkCudaErrors( cudaSetDevice( vtClrInfo[iCluster]->deviceId ) );

        // TODO: OPTIMIZATION: Device memories for weight data and neuron type map
        // must be allocated for each device; however, they can be
        // devided for each cluster.

        // allocate device memories for weight data 
        BGFLOAT* W_d;
        checkCudaErrors( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

        // and initialize it
        checkCudaErrors( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

        // allocate device memory for neuron type map
        neuronType* neuron_type_map_d;
        checkCudaErrors( cudaMalloc( ( void ** ) &neuron_type_map_d, sim_info->totalNeurons * sizeof( neuronType ) ) );

        // and initialize it
        checkCudaErrors( cudaMemcpy ( neuron_type_map_d, layout->neuron_type_map, sim_info->totalNeurons * sizeof( neuronType ), cudaMemcpyHostToDevice ) );

        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;
        int clusterNeuronsBegin = vtClrInfo[iCluster]->clusterNeuronsBegin;
        GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(vtClr[iCluster]);
        AllSpikingNeuronsDeviceProperties* allNeuronsDevice = GPUClr->m_allNeuronsDevice;
        AllSpikingSynapsesDeviceProperties* allSynapsesDevice = GPUClr->m_allSynapsesDevice;

        blocksPerGrid = ( totalClusterNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
        updateSynapsesWeightsDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, deltaT, W_d, sim_info->maxSynapsesPerNeuron, allNeuronsDevice, allSynapsesDevice, neuron_type_map_d, totalClusterNeurons, clusterNeuronsBegin );

        // copy device synapse count to host memory
        AllSynapses *synapses = dynamic_cast<AllSynapses*>(vtClr[iCluster]->m_synapses);
        synapses->copyDeviceSynapseCountsToHost(allSynapsesDevice, vtClrInfo[iCluster]);

        // copy device sourceNeuronLayoutIndex and in_use to host memory
        synapses->copyDeviceSourceNeuronIdxToHost(allSynapsesDevice, sim_info, vtClrInfo[iCluster]);

        // free device memories
        checkCudaErrors( cudaFree( W_d ) );
        checkCudaErrors( cudaFree( neuron_type_map_d ) );
    }

    // free host memories
    delete[] W_h;

    // Create synapse index maps
    SynapseIndexMap::createSynapseImap(sim_info, vtClr, vtClrInfo);
}
