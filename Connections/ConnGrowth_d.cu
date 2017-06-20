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
        AllSpikingNeurons *neurons = dynamic_cast<AllSpikingNeurons*>(vtClr[iCluster]->m_neurons);
        int neuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin;
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;

        // copy neuron's data from device memory to host
        GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(vtClr[iCluster]);
        neurons->copyNeuronDeviceSpikeCountsToHost(GPUClr->m_allNeuronsDevice, vtClrInfo[iCluster]);
        neurons->copyNeuronDeviceSpikeHistoryToHost(GPUClr->m_allNeuronsDevice, sim_info, vtClrInfo[iCluster]);

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

    // allocate device memories
    BGSIZE W_d_size = sim_info->totalNeurons * sim_info->totalNeurons * sizeof (BGFLOAT);
    BGFLOAT* W_h = new BGFLOAT[W_d_size];
    BGFLOAT* W_d;
    checkCudaErrors( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

    neuronType* neuron_type_map_d;
    checkCudaErrors( cudaMalloc( ( void ** ) &neuron_type_map_d, sim_info->totalNeurons * sizeof( neuronType ) ) );

    // copy weight data to the device memory
    for ( int i = 0 ; i < sim_info->totalNeurons; i++ )
        for ( int j = 0; j < sim_info->totalNeurons; j++ )
            W_h[i * sim_info->totalNeurons + j] = (*W)(i, j);

    checkCudaErrors( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMemcpy ( neuron_type_map_d, layout->neuron_type_map, sim_info->totalNeurons * sizeof( neuronType ), cudaMemcpyHostToDevice ) );

    // destination neurons of each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;
        int clusterNeuronsBegin = vtClrInfo[iCluster]->clusterNeuronsBegin;
        GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(vtClr[iCluster]);
        AllSpikingNeuronsDeviceProperties* allNeuronsDevice = GPUClr->m_allNeuronsDevice;
        AllSpikingSynapsesDeviceProperties* allSynapsesDevice = GPUClr->m_allSynapsesDevice;

        blocksPerGrid = ( totalClusterNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
        updateSynapsesWeightsDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, deltaT, W_d, sim_info->maxSynapsesPerNeuron, allNeuronsDevice, allSynapsesDevice, neuron_type_map_d, totalClusterNeurons, clusterNeuronsBegin );

        AllSynapses *synapses = dynamic_cast<AllSynapses*>(vtClr[iCluster]->m_synapses);
        // copy device synapse count to host memory
        synapses->copyDeviceSynapseCountsToHost(allSynapsesDevice, vtClrInfo[iCluster]);
        // copy device sourceNeuronLayoutIndex and in_use to host memory
        synapses->copyDeviceSourceNeuronIdxToHost(allSynapsesDevice, sim_info, vtClrInfo[iCluster]);
    }

    // free memories
    checkCudaErrors( cudaFree( W_d ) );
    delete[] W_h;

    checkCudaErrors( cudaFree( neuron_type_map_d ) );

    // Create synapse index maps
    SynapseIndexMap::createSynapseImap(sim_info, vtClr, vtClrInfo);

    // Copy synapse index maps to the device memory
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(vtClr[iCluster]);
        GPUClr->copySynapseIndexMapHostToDevice(vtClrInfo[iCluster]);
    }
}
