#include "ConnStatic.h"
#include "GPUSpikingCluster.h"
#include <helper_cuda.h>
#include <algorithm>
#include "HeapSort.hpp"

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
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid;

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
    AllSpikingNeuronsProps* allNeuronsDeviceProps = GPUClr->m_allNeuronsDeviceProps;
    AllSpikingSynapsesProps* allSynapsesDeviceProps = GPUClr->m_allSynapsesDeviceProps;

#ifdef PERFORMANCE_METRICS
    // Reset CUDA timer to start measurement of GPU operation
    cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

    blocksPerGrid = ( totalClusterNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
    setupConnectionsDevice <<< blocksPerGrid, threadsPerBlock >>> (GPUClr->m_synapsesDevice, num_neurons, totalClusterNeurons, clusterNeuronsBegin, xloc_d, yloc_d, m_nConnsPerNeuron, m_threshConnsRadius, neuron_type_map_d, rDistDestNeuron_d, sim_info->deltaT, allNeuronsDeviceProps, allSynapsesDeviceProps, m_excWeight[0], m_excWeight[1], m_inhWeight[0], m_inhWeight[1], devStates_d, time(NULL));

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
    AllSpikingSynapsesProps *pSynapsesProps = dynamic_cast<AllSpikingSynapsesProps*>(synapses->m_pSynapsesProps);
    pSynapsesProps->copyDeviceSynapseCountsToHost(allSynapsesDeviceProps, clr_info);

    // copy device sourceNeuronLayoutIndex and in_use to host memory
    pSynapsesProps->copyDeviceSourceNeuronIdxToHost(allSynapsesDeviceProps, sim_info, clr_info);

    // tell this thread's task has finished
    m_barrierSetupConnections->Sync();
} 

/*
 *  CUDA kernel function for setting up connections.
 *iNeuron  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters:
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param  synapsesDevice      Pointer to the Synapses object in device memory.
 *  @param  num_neurons         Number of total neurons.
 *  @param  totalClusterNeurons Total number of neurons in the cluster.
 *  @param  clusterNeuronsBegin Begin neuron index of the cluster.
 *  @param  xloc_d              Pointer to the neuron's x location array.
 *  @param  yloc_d              Pointer to the neuron's y location array.
 *  @param  nConnsPerNeuron     Number of maximum connections per neurons.
 *  @param  threshConnsRadius   Connection radius threshold.
 *  @param  neuron_type_map_d   Pointer to the neurons type map in device memory.
 *  @param  rDistDestNeuron_d   Pointer to the DistDestNeuron structure array.
 *  @param  deltaT              The time step size.
 *  @param  allNeuronsProps     Pointer to the Neuron structures in device memory.
 *  @param  allSynapsesProps    Pointer to the Synapse structures in device memory.
 *  @param  minExcWeight        Min values of excitatory neuron's synapse weight.
 *  @param  maxExcWeight        Max values of excitatory neuron's synapse weight.
 *  @param  minInhWeight        Min values of inhibitory neuron's synapse weight.
 *  @param  maxInhWeight        Max values of inhibitory neuron's synapse weight.
 *  @param  devStates_d         Curand global state.
 *  @param  seed                Seed for curand.
 */
__global__ void setupConnectionsDevice( IAllSynapses* synapsesDevice, int num_neurons, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* xloc_d, BGFLOAT* yloc_d, int nConnsPerNeuron, int threshConnsRadius, neuronType* neuron_type_map_d, ConnStatic::DistDestNeuron *rDistDestNeuron_d, BGFLOAT deltaT, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, BGFLOAT minExcWeight, BGFLOAT maxExcWeight, BGFLOAT minInhWeight, BGFLOAT maxInhWeight, curandState* devStates_d, unsigned long seed )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= totalClusterNeurons )
        return;

    int iNeuron = idx;
    int dest_neuron = iNeuron + clusterNeuronsBegin;

    // pick the connections shorter than threshConnsRadius
    BGSIZE iArrayBegin = num_neurons * iNeuron, iArrayEnd, iArray;
    iArray = iArrayBegin;
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        if (src_neuron != dest_neuron) {
            BGFLOAT distX = xloc_d[dest_neuron] - xloc_d[src_neuron];
            BGFLOAT distY = yloc_d[dest_neuron] - yloc_d[src_neuron];
            BGFLOAT dist2 = distX * distX + distY * distY;
            BGFLOAT dist = sqrt(dist2);

            if (dist <= threshConnsRadius) {
                ConnStatic::DistDestNeuron distDestNeuron;
                distDestNeuron.dist = dist;
                distDestNeuron.src_neuron = src_neuron;
                rDistDestNeuron_d[iArray++] = distDestNeuron;
            }
        }
    }

    // sort ascendant
    iArrayEnd = iArray;
    int size = iArrayEnd - iArrayBegin;
    // CUDA thrust sort consumes heap memory, and when sorting large contents
    // it may cause an error "temporary_buffer::allocate: get_temporary_buffer failed".
    // Therefore we use local implementation of heap sort.
    // NOTE: Heap sort is an in-palce algoprithm (memory requirement is 1).
    // Its implementation is not stable. Time complexity is O(n*logn).
    heapSort(&rDistDestNeuron_d[iArrayBegin], size);

    // set up an initial state for curand
    curand_init( seed, iNeuron, 0, &devStates_d[iNeuron] );

    // pick the shortest nConnsPerNeuron connections
    iArray = iArrayBegin;
    for (BGSIZE i = 0; iArray < iArrayEnd && (int)i < nConnsPerNeuron; iArray++, i++) {
        ConnStatic::DistDestNeuron distDestNeuron = rDistDestNeuron_d[iArray];
        int src_neuron = distDestNeuron.src_neuron;
        synapseType type = synapsesDevice->synType(neuron_type_map_d, src_neuron, dest_neuron);

        // create a synapse at the cluster of the destination neuron

        DEBUG_MID ( printf("source: %d dest: %d dist: %d\n", src_neuron, dest_neuron, distDestNeuron.dist); )

        // set synapse weight
        // TODO: we need another synaptic weight distibution mode (normal distribution)
        BGFLOAT weight;
        curandState localState = devStates_d[iNeuron];
        if (synapsesDevice->synSign(type) > 0) {
            weight = minExcWeight + curand_uniform( &localState ) * (maxExcWeight - minExcWeight);
        }
        else {
            weight = minInhWeight + curand_uniform( &localState ) * (maxInhWeight - minInhWeight);
        }
        devStates_d[iNeuron] = localState;

        BGFLOAT* sum_point = &( allNeuronsProps->summation_map[iNeuron] );
        BGSIZE iSyn;
        synapsesDevice->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, deltaT, iNeuron);
        allSynapsesProps->W[iSyn] = weight;

    }
}

