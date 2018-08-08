#include "ConnStatic.h"
#include "GPUSpikingCluster.h"
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
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid;

    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    int num_neurons = sim_info->totalNeurons;
    int totalClusterNeurons = clr_info->totalClusterNeurons;
    int clusterNeuronsBegin = clr_info->clusterNeuronsBegin;

    // allocate host memory for distance between neurons
    vector<DistDestNeuron> h_distDestNeurons;
    BGFLOAT* dist_h = new BGFLOAT[num_neurons];
    
    // allocate GPU memory for distance between neurons
    BGFLOAT* dist_d;
    checkCudaErrors( cudaMalloc( ( void **) &dist_d, num_neurons * sizeof (BGFLOAT) ) );

    // allocate device memory for neuron's location data and initialize it
    BGFLOAT* xloc_d;
    BGFLOAT* yloc_d;
    checkCudaErrors( cudaMalloc( ( void ** ) &xloc_d, num_neurons * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &yloc_d, num_neurons * sizeof( BGFLOAT ) ) );

    checkCudaErrors( cudaMemcpy ( xloc_d, layout->xloc, num_neurons * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( yloc_d, layout->yloc, num_neurons * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );

    int added = 0;

    // for each destination neuron in the cluster
    for (int dest_neuron = clusterNeuronsBegin, iNeuron = 0; iNeuron < totalClusterNeurons; dest_neuron++, iNeuron++) {
        h_distDestNeurons.clear();

        // calculate the distance between neurons
        blocksPerGrid = ( num_neurons + threadsPerBlock - 1 ) / threadsPerBlock;
        calcNeuronsDistanceDevice <<< blocksPerGrid, threadsPerBlock >>> ( num_neurons, dest_neuron, xloc_d, yloc_d, dist_d );
        
        // copy distance data from device to host
        checkCudaErrors( cudaMemcpy ( dist_h, dist_d, num_neurons * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );

        // pick the connections shorter than threshConnsRadius
        for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
            if (src_neuron != dest_neuron) {
                if (dist_h[src_neuron] <= m_threshConnsRadius) {
                    DistDestNeuron distDestNeuron;
                    distDestNeuron.dist = dist_h[src_neuron];
                    distDestNeuron.src_neuron = src_neuron;
                    h_distDestNeurons.push_back(distDestNeuron);
                }
            }
        }

        // sort ascendant
        sort(h_distDestNeurons.begin(), h_distDestNeurons.end());

        // pick the shortest m_nConnsPerNeuron connections
        for (BGSIZE i = 0; i < h_distDestNeurons.size() && (int)i < m_nConnsPerNeuron; i++) {
            int src_neuron = h_distDestNeurons[i].src_neuron;
            synapseType type = layout->synType(src_neuron, dest_neuron);

            // create a synapse at the cluster of the destination neuron
            IAllNeurons *neurons = clr->m_neurons;
            IAllSynapses *synapses = clr->m_synapses;

            DEBUG_MID (cout << "source: " << src_neuron << " dest: " << dest_neuron << " dist: " << h_distDestNeurons[i].dist << endl;)

            BGFLOAT* sum_point = &( dynamic_cast<AllNeurons*>(neurons)->summation_map[iNeuron] );
            BGSIZE iSyn;
            synapses->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, sim_info->deltaT, clr_info);
            added++;

            // set synapse weight
            // TODO: we need another synaptic weight distibution mode (normal distribution)
            if (synapses->synSign(type) > 0) {
                dynamic_cast<AllSynapses*>(synapses)->W[iSyn] = rng.inRange(m_excWeight[0], m_excWeight[1]);
            }
            else {
                dynamic_cast<AllSynapses*>(synapses)->W[iSyn] = rng.inRange(m_inhWeight[0], m_inhWeight[1]);
            }
        }
    }

    // free memories
    delete[] dist_h;

    checkCudaErrors( cudaFree( dist_d ) );
    checkCudaErrors( cudaFree( xloc_d ) );
    checkCudaErrors( cudaFree( yloc_d ) );

    int nRewiring = added * m_pRewiring;

    DEBUG(cout << "Rewiring connections: " << nRewiring << endl;)

    DEBUG (cout << "added connections: " << added << endl << endl << endl;)

    // copy host synapse arrays into GPU device
    AllSynapses *synapses = dynamic_cast<AllSynapses*>(clr->m_synapses);
    AllSpikingSynapsesDeviceProperties* allSynapsesDevice = dynamic_cast<GPUSpikingCluster*>(clr)->m_allSynapsesDevice;
    synapses->copySynapseHostToDevice( allSynapsesDevice, sim_info, clr_info );

    // tell this thread's task has finished
    m_barrierSetupConnections->Sync();
} 

/*
 *  CUDA kernel function for calculating distance between n eurons.
 *
 *  @param  num_neurons      Number of total neurons.
 *  @param  dest_neuron      Destination neuron layout index.
 *  @param  xloc_d           Neurons x locations.
 *  @param  yloc_d           Neurons y locations.
 *  @param  dist_d           Pointer to the array where results are stored.
 */
__global__ void calcNeuronsDistanceDevice( int num_neurons, int dest_neuron, BGFLOAT* xloc_d, BGFLOAT* yloc_d, BGFLOAT* dist_d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_neurons )
        return;

    int src_neuron = idx;

    BGFLOAT distX = xloc_d[dest_neuron] - xloc_d[src_neuron];
    BGFLOAT distY = yloc_d[dest_neuron] - yloc_d[src_neuron];
    BGFLOAT dist2 = distX * distX + distY * distY;
    dist_d[src_neuron] = sqrt(dist2);
}
