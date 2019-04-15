#include "ConnStatic.h"
#include "GPUSpikingCluster.h"
#include "AllSynapsesDeviceFuncs.h"
#include <helper_cuda.h>
#include <algorithm>

//  Initialize the Barrier Synchnonize object for setup connections.
Barrier *ConnStatic::m_barrierSetupConnections = NULL;

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters: 
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  vtClr       Vector of Cluster class objects.
 *  @param  vtClrInfo   Vector of ClusterInfo.
 */
void ConnStatic::setupConnections(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    DEBUG(cout << "Initializing connections" << endl;)

    // Initialize the Barrier Synchnonize object for setupConnections.
    m_barrierSetupConnections = new Barrier(vtClr.size() + 1);

    // for each cluster
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        // create an setupConnectionsThread
        std::thread thSetupConnectionsThread(&ConnStatic::setupConnectionsThread, this, sim_info, layout, vtClr[iCluster], vtClrInfo[iCluster]);

        // leave it running
        thSetupConnectionsThread.detach();
    }

    // wait until every setupConnectionsThread finishes
    m_barrierSetupConnections->Sync();

    // Create synapse index maps
    SynapseIndexMap::createSynapseImap(sim_info, vtClr, vtClrInfo);

    delete m_barrierSetupConnections;
}

/*
 *  Thread for setting the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  layout      Layout information of the neunal network.
 *  @param  clr         Pointer to cluster class to read information from.
 *  @param  clr_info    Pointer to clusterInfo class to read information from.
 */ 
void ConnStatic::setupConnectionsThread(const SimulationInfo *sim_info, Layout *layout, Cluster * clr, ClusterInfo * clr_info)
{
    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    int num_neurons = sim_info->totalNeurons;
    int totalClusterNeurons = clr_info->totalClusterNeurons;
    int clusterNeuronsBegin = clr_info->clusterNeuronsBegin;

    // allocate device memory for neuron type map
    neuronType* neuron_type_map_d;
    checkCudaErrors( cudaMalloc( ( void ** ) &neuron_type_map_d, sim_info->totalNeurons * sizeof( neuronType ) ) );

    // and initialize it
    checkCudaErrors( cudaMemcpy ( neuron_type_map_d, layout->neuron_type_map, sim_info->totalNeurons * sizeof( neuronType ), cudaMemcpyHostToDevice ) );

    // allocate device memory for neuron's location data and initialize it
    BGFLOAT* xloc_d;
    BGFLOAT* yloc_d;
    checkCudaErrors( cudaMalloc( ( void ** ) &xloc_d, num_neurons * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &yloc_d, num_neurons * sizeof( BGFLOAT ) ) );

    checkCudaErrors( cudaMemcpy ( xloc_d, layout->xloc, num_neurons * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( yloc_d, layout->yloc, num_neurons * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );

    // allocate memory for curand global state & setup it
    curandState* devStates_d;
    checkCudaErrors( cudaMalloc ( &(devStates_d), totalClusterNeurons * sizeof( curandState ) ) );

    // allocate working memory for setup connections
    DistDestNeuron *rDistDestNeuron_d;
    checkCudaErrors( cudaMalloc ( &(rDistDestNeuron_d), num_neurons * totalClusterNeurons * sizeof( DistDestNeuron ) ) );
     
    int added = 0;

    // for each destination neuron in the cluster
    GPUSpikingCluster *GPUClr = dynamic_cast<GPUSpikingCluster *>(clr);
    AllSpikingNeuronsDeviceProperties* allNeuronsDevice = GPUClr->m_allNeuronsDevice;
    AllSpikingSynapsesDeviceProperties* allSynapsesDevice = GPUClr->m_allSynapsesDevice;

#ifdef PERFORMANCE_METRICS
    // Reset CUDA timer to start measurement of GPU operation
    cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

    setupConnectionsDevice <<< clr_info->neuronBlocksPerGrid, clr_info->threadsPerBlock >>> (num_neurons, totalClusterNeurons, clusterNeuronsBegin, xloc_d, yloc_d, m_nConnsPerNeuron, m_threshConnsRadius, neuron_type_map_d, rDistDestNeuron_d, sim_info->deltaT, allNeuronsDevice, allSynapsesDevice, m_excWeight[0], m_excWeight[1], m_inhWeight[0], m_inhWeight[1], devStates_d, time(NULL));

#ifdef PERFORMANCE_METRICS
    cudaLapTime(clr_info, clr_info->t_gpu_setupConns);
#endif // PERFORMANCE_METRICS

    // free device memories
    checkCudaErrors( cudaFree( rDistDestNeuron_d ) );
    checkCudaErrors( cudaFree( devStates_d ) );
    checkCudaErrors( cudaFree( neuron_type_map_d ) );
    checkCudaErrors( cudaFree( xloc_d ) );
    checkCudaErrors( cudaFree( yloc_d ) );

    // TODO: need to implement rewiring
    int nRewiring = added * m_pRewiring;

    DEBUG(cout << "Rewiring connections: " << nRewiring << endl;)

    DEBUG (cout << "added connections: " << added << endl << endl << endl;)

    // copy device synapse count to host memory
    AllSynapses *synapses = dynamic_cast<AllSynapses*>(clr->m_synapses);
    synapses->copyDeviceSynapseCountsToHost(allSynapsesDevice, clr_info);

    // copy device sourceNeuronLayoutIndex and in_use to host memory
    synapses->copyDeviceSourceNeuronIdxToHost(allSynapsesDevice, sim_info, clr_info);

    // tell this thread's task has finished
    m_barrierSetupConnections->Sync();
} 
