#include "ConnGrowth.h"
#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include "GPUSpikingCluster.h"
#include <helper_cuda.h>

//  Initialize the Barrier Synchnonize object for update connections.
Barrier *ConnGrowth::m_barrierUpdateConnections = NULL;

/*
 *  Update the connections status in every epoch.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  vtClr       Vector of Cluster class objects.
 *  @param  vtClrInfo   Vector of ClusterInfo.
 */
void ConnGrowth::updateConnections(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    // Update Connections data
    updateConns(sim_info, vtClr, vtClrInfo);

    // wait until every updateConnsThread finishes
    m_barrierUpdateConnections->Sync();

    // Update the weight of the synapses
    updateSynapsesWeights(sim_info, layout, vtClr, vtClrInfo);

    // wait until every updateSynapsesWeightsThread finishes
    m_barrierUpdateConnections->Sync();

    // Create synapse index maps
    SynapseIndexMap::createSynapseImap(sim_info, vtClr, vtClrInfo);
}

/*
 *  Calculates firing rates, neuron radii change and assign new values.
 *  Creates a thread for each cluster and transfer the task.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  vtClr       Vector of Cluster class objects.
 *  @param  vtClrInfo   Vector of ClusterInfo.
 */
void ConnGrowth::updateConns(const SimulationInfo *sim_info, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        // create an updateConnsThread
        std::thread thUpdateConns(&ConnGrowth::updateConnsThread, this, sim_info, vtClr[iCluster], vtClrInfo[iCluster]);

        // leave it running
        thUpdateConns.detach();
    }
}

/*
 *  Thread for calculating firing rates, neuron radii change and assign new values.
 *  Executes a CUDA kernel function to do the task.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr         Pointer to cluster class to read information from.
 *  @param  clr_info    Pointer to clusterInfo class to read information from.
 */
void ConnGrowth::updateConnsThread(const SimulationInfo *sim_info, Cluster *clr, ClusterInfo *clr_info)
{
    int blockPerGrid = clr_info->blocksPerGrid;
    int threadsPerBlock = clr_info->threadsPerBlock;

    // pointer to the GPU global memory to save rates and radii
    BGFLOAT* rates_d;
    BGFLOAT* radii_d;

    // Set device ID
    checkCudaErrors(cudaSetDevice(clr_info->deviceId));

    // Calculate growth cycle firing rate for previous period
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);

    int totalClusterNeurons = clr_info->totalClusterNeurons;
    int clusterNeuronsBegin = clr_info->clusterNeuronsBegin;

    // allocates GPU memories
    checkCudaErrors(cudaMalloc((void **)&rates_d, totalClusterNeurons * sizeof(BGFLOAT)));
    checkCudaErrors(cudaMalloc((void **)&radii_d, totalClusterNeurons * sizeof(BGFLOAT)));

    // copy radii data from host to device
    checkCudaErrors(cudaMemcpy(radii_d, &radii[clusterNeuronsBegin], totalClusterNeurons * sizeof(BGFLOAT), cudaMemcpyHostToDevice));

    // executes a CUDA kernel function
    GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(clr);
    AllSpikingNeuronsDeviceProperties* allNeuronsDevice = GPUClr->m_allNeuronsDevice;

#ifdef PERFORMANCE_METRICS
    // Reset CUDA timer to start measurement of GPU operation
    cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

    // CUDA kernel function for calculating firing rates, neuron radii change and assign new values
    updateConnsDevice <<< blockPerGrid, threadsPerBlock >>> (allNeuronsDevice, totalClusterNeurons, max_spikes, sim_info->epochDuration, m_growth.maxRate, m_growth.beta, m_growth.rho, m_growth.epsilon, rates_d, radii_d);

#ifdef PERFORMANCE_METRICS
    cudaLapTime(clr_info, clr_info->t_gpu_updateConns);
#endif // PERFORMANCE_METRICS

    // copy rates and radii data back to host
    checkCudaErrors(cudaMemcpy(&rates[clusterNeuronsBegin], rates_d, totalClusterNeurons * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&radii[clusterNeuronsBegin], radii_d, totalClusterNeurons * sizeof(BGFLOAT), cudaMemcpyDeviceToHost));

    // free GPU memories
    checkCudaErrors(cudaFree(rates_d));
    checkCudaErrors(cudaFree(radii_d));

    // tell this thread's task has finished
    m_barrierUpdateConnections->Sync();
}

/*
 *  Update the weight of the Synapses in the simulation.
 *  Creates a thread for each cluster and transfer the task.
 *
 *  @param  sim_info    SimulationInfo to refer from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  vtClr       Vector of Cluster class objects.
 *  @param  vtClrInfo   Vector of ClusterInfo.
 */
void ConnGrowth::updateSynapsesWeights(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        // create an updateSynapsesWeightsThread
        std::thread thUpdateSynapsesWeights(&ConnGrowth::updateSynapsesWeightsThread, this, sim_info, layout, vtClr[iCluster], vtClrInfo[iCluster]);

        // leave it running
        thUpdateSynapsesWeights.detach();
    }
}

/*
 *  Thread for Updating the weight of the Synapses in the simulation.
 *  Executes a CUDA kernel function to do the task.
 *
 *  @param  sim_info    SimulationInfo to refer from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  clr         Pointer to cluster class to read information from.
 *  @param  clr_info    Pointer to clusterInfo class to read information from.
 */
void ConnGrowth::updateSynapsesWeightsThread(const SimulationInfo *sim_info, Layout *layout, Cluster *clr, ClusterInfo *clr_info)
{
    int blockPerGrid = clr_info->blocksPerGrid;
    int threadsPerBlock = clr_info->threadsPerBlock;

    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).

    BGFLOAT deltaT = sim_info->deltaT;

    // pointer to the GPU global memory to save radii
    BGFLOAT* radii_d;

    // Set device ID
    checkCudaErrors(cudaSetDevice(clr_info->deviceId));

    int totalClusterNeurons = clr_info->totalClusterNeurons;
    int clusterNeuronsBegin = clr_info->clusterNeuronsBegin;

    // allocate GPU memory and copy data from host to device
    checkCudaErrors(cudaMalloc((void **)&radii_d, sim_info->totalNeurons * sizeof(BGFLOAT)));
    checkCudaErrors(cudaMemcpy(radii_d, radii, sim_info->totalNeurons * sizeof(BGFLOAT), cudaMemcpyHostToDevice));

    // allocate device memory for neuron type map
    neuronType* neuron_type_map_d;
    checkCudaErrors(cudaMalloc((void **)&neuron_type_map_d, sim_info->totalNeurons * sizeof(neuronType)));

    // and initialize it
    checkCudaErrors(cudaMemcpy(neuron_type_map_d, layout->neuron_type_map, sim_info->totalNeurons * sizeof(neuronType), cudaMemcpyHostToDevice));

    // allocate device memory for neuron's location data and initialize it
    BGFLOAT* xloc_d;
    BGFLOAT* yloc_d;
    checkCudaErrors(cudaMalloc((void **)&xloc_d, sim_info->totalNeurons * sizeof(BGFLOAT)));
    checkCudaErrors(cudaMalloc((void **)&yloc_d, sim_info->totalNeurons * sizeof(BGFLOAT)));

    checkCudaErrors(cudaMemcpy(xloc_d, layout->xloc, sim_info->totalNeurons * sizeof(BGFLOAT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(yloc_d, layout->yloc, sim_info->totalNeurons * sizeof(BGFLOAT), cudaMemcpyHostToDevice));

    // executes a CUDA kernel function
    GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(clr);
    AllSpikingNeuronsDeviceProperties* allNeuronsDevice = GPUClr->m_allNeuronsDevice;
    AllSpikingSynapsesDeviceProperties* allSynapsesDevice = GPUClr->m_allSynapsesDevice;

#ifdef PERFORMANCE_METRICS
    // Reset CUDA timer to start measurement of GPU operation
    cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

    updateSynapsesWeightsDevice <<< blocksPerGrid, threadsPerBlock >>> (sim_info->totalNeurons, deltaT, sim_info->maxSynapsesPerNeuron, allNeuronsDevice, allSynapsesDevice, neuron_type_map_d, totalClusterNeurons, clusterNeuronsBegin, radii_d, xloc_d, yloc_d);

#ifdef PERFORMANCE_METRICS
    cudaLapTime(clr_info, clr_info->t_gpu_updateSynapsesWeights);
#endif // PERFORMANCE_METRICS

    // copy device synapse count to host memory
    AllSynapses *synapses = dynamic_cast<AllSynapses*>(clr->m_synapses);
    synapses->copyDeviceSynapseCountsToHost(allSynapsesDevice, clr_info);

    // copy device sourceNeuronLayoutIndex and in_use to host memory
    synapses->copyDeviceSourceNeuronIdxToHost(allSynapsesDevice, sim_info, clr_info);

    DEBUG(
        {
            // Report GPU memory usage
            printf("\n");

            size_t free_byte;
            size_t total_byte;

            checkCudaErrors(cudaMemGetInfo(&free_byte, &total_byte));

            double free_db = (double)free_byte;
            double total_db = (double)total_byte;
            double used_db = total_db - free_db;

            printf("Updating Synapse Weights\n");
            printf("GPU memory usage: device ID = %d, used = %5.3f MB, free = %5.3f MB, total = %5.3f MB\n", clr_info->deviceId, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

            printf("\n");
        }
    ); // end  DEBUG

      // free device memories
    checkCudaErrors(cudaFree(neuron_type_map_d));

    checkCudaErrors(cudaFree(xloc_d));
    checkCudaErrors(cudaFree(yloc_d));

    checkCudaErrors(cudaFree(radii_d));

    // tell this thread's task has finished
    m_barrierUpdateConnections->Sync();
}

/*
 *  CUDA kernel function for calculating firing rates, neuron radii change and assign new values.
 *
 *  @param  allNeuronsDevice       Pointer to Neuron structures in device memory.
 *  @param  totalClusterNeurons    Number of neurons in the cluster.
 *  @param  max_spikes             Maximum firing rate.
 *  @param  epochDuration          One epoch duration in second.
 *  @param  maxRate                Growth parameter (= targetRate / epsilon)
 *  @param  beta                   Growth parameter (sensitivity of outgrowth to firing rate)
 *  @param  rho                    Growth parameter (outgrowth rate constant)
 *  @param  epsilona               Growth parameter (null firing rate(zero outgrowth))
 *  @param  rates_d                Pointer to rates data array.
 *  @param  radii_d                Pointer to radii data array.
 */
__global__ void updateConnsDevice(AllSpikingNeuronsDeviceProperties* allNeuronsDevice, int totalClusterNeurons, int max_spikes, BGFLOAT epochDuration, BGFLOAT maxRate, BGFLOAT beta, BGFLOAT rho, BGFLOAT epsilon, BGFLOAT* rates_d, BGFLOAT* radii_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalClusterNeurons)
        return;

    int iNeuron = idx;

    DEBUG(assert(allNeuronsDevice->spikeCount[iNeuron] < max_spikes); );

    // compute neuron radii change and assign new values
    rates_d[iNeuron] = allNeuronsDevice->spikeCount[iNeuron] / epochDuration;
    BGFLOAT outgrowth = 1.0 - 2.0 / (1.0 + exp(static_cast<double>((epsilon - rates_d[iNeuron] / maxRate)) / beta));
    BGFLOAT deltaR = epochDuration * rho * outgrowth;
    radii_d[iNeuron] += deltaR;
}

