#include "ConnGrowth.h"
#include "AllSpikingSynapses.h"
#include "GPUSpikingCluster.h"
#include <helper_cuda.h>
#include "math_constants.h"

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
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blockPerGrid;

    // pointer to the GPU global memory to save rates and radii
    BGFLOAT* rates_d;
    BGFLOAT* radii_d;

    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    // Calculate growth cycle firing rate for previous period
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);

    int totalClusterNeurons = clr_info->totalClusterNeurons;
    int clusterNeuronsBegin = clr_info->clusterNeuronsBegin;

    // allocates GPU memories
    checkCudaErrors( cudaMalloc( ( void **) &rates_d, totalClusterNeurons * sizeof (BGFLOAT) ) );
    checkCudaErrors( cudaMalloc( ( void **) &radii_d, totalClusterNeurons * sizeof (BGFLOAT) ) );

    // copy radii data from host to device
    checkCudaErrors( cudaMemcpy ( radii_d, &radii[clusterNeuronsBegin], totalClusterNeurons * sizeof (BGFLOAT), cudaMemcpyHostToDevice ) );

    // executes a CUDA kernel function
    GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(clr);
    AllSpikingNeuronsProps* allNeuronsProps = GPUClr->m_allNeuronsDeviceProps;      

#ifdef PERFORMANCE_METRICS
    // Reset CUDA timer to start measurement of GPU operation
    cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

    // CUDA kernel function for calculating firing rates, neuron radii change and assign new values
    blockPerGrid = ( totalClusterNeurons + threadsPerBlock - 1) / threadsPerBlock;
    updateConnsDevice <<< blockPerGrid, threadsPerBlock >>> (allNeuronsProps, totalClusterNeurons, max_spikes, sim_info->epochDuration, m_growth.maxRate, m_growth.beta, m_growth.rho, m_growth.epsilon, rates_d, radii_d );

#ifdef PERFORMANCE_METRICS
    cudaLapTime(clr_info, clr_info->t_gpu_updateConns);
#endif // PERFORMANCE_METRICS

    // copy rates and radii data back to host
    checkCudaErrors( cudaMemcpy ( &rates[clusterNeuronsBegin], rates_d, totalClusterNeurons * sizeof (BGFLOAT), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( &radii[clusterNeuronsBegin], radii_d, totalClusterNeurons * sizeof (BGFLOAT), cudaMemcpyDeviceToHost ) );

    // free GPU memories
    checkCudaErrors( cudaFree( rates_d ) );
    checkCudaErrors( cudaFree( radii_d ) );

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
    // For now, we just set the weights to equal the areas. We will later
    // scale it and set its sign (when we index and get its sign).

    BGFLOAT deltaT = sim_info->deltaT;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid;

    // pointer to the GPU global memory to save radii
    BGFLOAT* radii_d;

    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    int totalClusterNeurons = clr_info->totalClusterNeurons;
    int clusterNeuronsBegin = clr_info->clusterNeuronsBegin;

    // allocate GPU memory and copy data from host to device
    checkCudaErrors( cudaMalloc( ( void **) &radii_d, sim_info->totalNeurons * sizeof (BGFLOAT) ) );
    checkCudaErrors( cudaMemcpy ( radii_d, radii, sim_info->totalNeurons * sizeof (BGFLOAT), cudaMemcpyHostToDevice ) );

    // allocate device memory for neuron type map
    neuronType* neuron_type_map_d;
    checkCudaErrors( cudaMalloc( ( void ** ) &neuron_type_map_d, sim_info->totalNeurons * sizeof( neuronType ) ) );

    // and initialize it
    checkCudaErrors( cudaMemcpy ( neuron_type_map_d, layout->neuron_type_map, sim_info->totalNeurons * sizeof( neuronType ), cudaMemcpyHostToDevice ) );

    // allocate device memory for neuron's location data and initialize it
    BGFLOAT* xloc_d;
    BGFLOAT* yloc_d;
    checkCudaErrors( cudaMalloc( ( void ** ) &xloc_d, sim_info->totalNeurons * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &yloc_d, sim_info->totalNeurons * sizeof( BGFLOAT ) ) );

    checkCudaErrors( cudaMemcpy ( xloc_d, layout->xloc, sim_info->totalNeurons * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( yloc_d, layout->yloc, sim_info->totalNeurons * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );

    // executes a CUDA kernel function
    GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(clr);
    AllSpikingNeuronsProps* allNeuronsDeviceProps = GPUClr->m_allNeuronsDeviceProps;
    AllSpikingSynapsesProps* allSynapsesDeviceProps = GPUClr->m_allSynapsesDeviceProps;

#ifdef PERFORMANCE_METRICS
    // Reset CUDA timer to start measurement of GPU operation
    cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

    blocksPerGrid = ( totalClusterNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
    updateSynapsesWeightsDevice <<< blocksPerGrid, threadsPerBlock >>> ( GPUClr->m_synapsesDevice, sim_info->totalNeurons, deltaT, sim_info->maxSynapsesPerNeuron, allNeuronsDeviceProps, allSynapsesDeviceProps, neuron_type_map_d, totalClusterNeurons, clusterNeuronsBegin, radii_d, xloc_d, yloc_d );

#ifdef PERFORMANCE_METRICS
    cudaLapTime(clr_info, clr_info->t_gpu_updateSynapsesWeights);
#endif // PERFORMANCE_METRICS

    // copy device synapse count to host memory
    AllSynapses *synapses = dynamic_cast<AllSynapses*>(clr->m_synapses);
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(synapses->m_pSynapsesProps);
    pSynapsesProps->copyDeviceSynapseCountsToHost(allSynapsesDeviceProps, clr_info);

    // copy device sourceNeuronLayoutIndex and in_use to host memory
    pSynapsesProps->copyDeviceSourceNeuronIdxToHost(allSynapsesDeviceProps, sim_info, clr_info);

  DEBUG(
    {
        // Report GPU memory usage
        printf("\n");

        size_t free_byte;
        size_t total_byte;

        checkCudaErrors( cudaMemGetInfo( &free_byte, &total_byte ) );

        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db = total_db - free_db;

        printf("Updating Synapse Weights\n");
        printf("GPU memory usage: device ID = %d, used = %5.3f MB, free = %5.3f MB, total = %5.3f MB\n", clr_info->deviceId, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

        printf("\n");
    }
  ) // end  DEBUG

    // free device memories
    checkCudaErrors( cudaFree( neuron_type_map_d ) );

    checkCudaErrors( cudaFree( xloc_d ) );
    checkCudaErrors( cudaFree( yloc_d ) );

    checkCudaErrors( cudaFree( radii_d ) );

    // tell this thread's task has finished
    m_barrierUpdateConnections->Sync();
}

/*
 *  CUDA kernel function for calculating firing rates, neuron radii change and assign new values.
 *
 *  @param  allNeuronsProps   Pointer to Neuron structures in device memory.
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
__global__ void updateConnsDevice( AllSpikingNeuronsProps* allNeuronsProps, int totalClusterNeurons, int max_spikes, BGFLOAT epochDuration, BGFLOAT maxRate, BGFLOAT beta, BGFLOAT rho, BGFLOAT epsilon, BGFLOAT* rates_d, BGFLOAT* radii_d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= totalClusterNeurons )
        return;

    int iNeuron = idx;

    DEBUG( assert(allNeuronsProps->spikeCount[iNeuron] < max_spikes); )

    // compute neuron radii change and assign new values
    rates_d[iNeuron] = allNeuronsProps->spikeCount[iNeuron] / epochDuration;
    BGFLOAT outgrowth = 1.0 - 2.0 / (1.0 + exp(static_cast<double>((epsilon - rates_d[iNeuron] / maxRate)) / beta));
    BGFLOAT deltaR = epochDuration * rho * outgrowth;
    radii_d[iNeuron] += deltaR;
}

/* -------------------------------------*\
|* # Global Functions for updateSynapses
\* -------------------------------------*/

/*
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below
 * zero.
 *
 * @param[in] synapsesDevice     Pointer to the Synapses object in device memory.
 * @param[in] num_neurons        Total number of neurons.
 * @param[in] deltaT             The time step size.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsProps    Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesProps   Pointer to the Synapse structures in device memory.
 * @param[in] neuron_type_map_d    Pointer to the neurons type map in device memory.
 * @param[in] totalClusterNeurons  Total number of neurons in the cluster.
 * @param[in] clusterNeuronsBegin  Begin neuron index of the cluster.
 * @param[in] radii_d            Pointer to the rates data array.
 * @param[in] xloc_d             Pointer to the neuron's x location array.
 * @param[in] yloc_d             Pointer to the neuron's y location array.
 */
__global__ void updateSynapsesWeightsDevice( IAllSynapses* synapsesDevice, int num_neurons, BGFLOAT deltaT, int maxSynapses, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, neuronType* neuron_type_map_d, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* radii_d, BGFLOAT* xloc_d,  BGFLOAT* yloc_d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= totalClusterNeurons )
        return;

    int adjusted = 0;
    int removed = 0;
    int added = 0;

    int iNeuron = idx;
    int dest_neuron = clusterNeuronsBegin + idx;

    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        if (dest_neuron == src_neuron) {
            // we don't create a synapse between the same neuron.
            continue;
        }

        // Update the areas of overlap in between Neurons
        BGFLOAT distX = xloc_d[dest_neuron] - xloc_d[src_neuron];
        BGFLOAT distY = yloc_d[dest_neuron] - yloc_d[src_neuron];
        BGFLOAT dist2 = distX * distX + distY * distY;
        BGFLOAT dist = sqrt(dist2);
        BGFLOAT delta = dist - (radii_d[dest_neuron] + radii_d[src_neuron]);
        BGFLOAT area = 0.0;

        if (delta < 0) {
            BGFLOAT lenAB = dist;
            BGFLOAT r1 = radii_d[dest_neuron];
            BGFLOAT r2 = radii_d[src_neuron];

            if (lenAB + min(r1, r2) <= max(r1, r2)) {
                area = CUDART_PI_F * min(r1, r2) * min(r1, r2); // Completely overlapping unit
            } else {
                // Partially overlapping unit
                BGFLOAT lenAB2 = dist2;
                BGFLOAT r12 = r1 * r1;
                BGFLOAT r22 = r2 * r2;

                BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);

                if(fabs(cosCBA) >= 1.0 || fabs(cosCAB) >= 1.0) {
                    area = 0.0;
                } else {
                    BGFLOAT angCBA = acos(cosCBA);
                    BGFLOAT angCBD = 2.0 * angCBA;

                    BGFLOAT angCAB = acos(cosCAB);
                    BGFLOAT angCAD = 2.0 * angCAB;

                    area = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
                }
            }
        }

        // visit each synapse at (xa,ya)
        bool connected = false;
        synapseType type = synapsesDevice->synType(neuron_type_map_d, src_neuron, dest_neuron);

        // for each existing synapse
        BGSIZE existing_synapses = allSynapsesProps->synapse_counts[iNeuron];
        int existing_synapses_checked = 0;
        for (BGSIZE synapse_index = 0; (existing_synapses_checked < existing_synapses) && !connected; synapse_index++) {
            BGSIZE iSyn = maxSynapses * iNeuron + synapse_index;
            if (allSynapsesProps->in_use[iSyn] == true) {
                // if there is a synapse between a and b
                if (allSynapsesProps->sourceNeuronLayoutIndex[iSyn] == src_neuron) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.

                    // W_d[] is indexed by (dest_neuron (local index) * totalNeurons + src_neuron)
                    if (area <= 0) {
                        removed++;
                        synapsesDevice->eraseSynapse(iNeuron, iSyn);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesProps->W[iSyn] = area * synapsesDevice->synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
                    }
                }
                existing_synapses_checked++;
            }
        }

        // if not connected and weight(a,b) > 0, add a new synapse from a to b
        if (!connected && (area > 0)) {
            // locate summation point
            BGFLOAT* sum_point = &( allNeuronsProps->summation_map[iNeuron] );
            added++;

            BGSIZE iSyn;
            synapsesDevice->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, deltaT, iNeuron);
            BGFLOAT weight = area * synapsesDevice->synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
            allSynapsesProps->W[iSyn] = weight;
        }
    }
}

