#include "AllSpikingNeuronsProperties.h"

#if !defined(USE_GPU)

// Default constructor
AllSpikingNeuronsProperties::AllSpikingNeuronsProperties()
{
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

AllSpikingNeuronsProperties::~AllSpikingNeuronsProperties()
{
    cleanupNeuronsProperties();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingNeuronsProperties::setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllNeuronsProperties::setupNeuronsProperties(sim_info, clr_info);

    // TODO: Rename variables for easier identification
    hasFired = new bool[size];
    spikeCount = new int[size];
    spikeCountOffset = new int[size];
    spike_history = new uint64_t*[size];

    for (int i = 0; i < size; ++i) {
        spike_history[i] = NULL;
        hasFired[i] = false;
        spikeCount[i] = 0;
        spikeCountOffset[i] = 0;
    }
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSpikingNeuronsProperties::cleanupNeuronsProperties()
{
    if (size != 0) {
        for(int i = 0; i < size; i++) {
            delete[] spike_history[i];
        }

        delete[] hasFired;
        delete[] spikeCount;
        delete[] spikeCountOffset;
        delete[] spike_history;
    }

    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

#else // USE_GPU

// Default constructor
AllSpikingNeuronsProperties::AllSpikingNeuronsProperties()
{
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

AllSpikingNeuronsProperties::~AllSpikingNeuronsProperties()
{
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  pAllNeuronsProperties_d the AllNeuronsProperties on device memory.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
__host__ static void AllSpikingNeuronsProperties::setupNeuronsProperties(void *pAllNeuronsProperties_d, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingNeuronsProperties allNeuronsProperties;

    // allocate GPU memories to store all neuron's states
    allocNeuronsProperties(allNeuronsProperties, sim_info, clr_info);

    // copy the pointer address to structure on device memory
    checkCudaErrors( cudaMemcpy ( pAllNeuronsDeviceProperties_d, &allNeuronsProperties, sizeof( AllSpikingNeuronsProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProperties   Reference to the AllSpikingNeuronsProperties struct.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllSpikingNeuronsProperties::allocNeuronsProperties(AllSpikingNeuronsProperties &allNeuronsProperties, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);

    AllNeuronsProperties::allocNeuronsProperties(allNeuronsProperties, clr_info);

    // allocate GPU memories to store all neuron's states
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.hasFired, size * sizeof( bool ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.spikeCount, size * sizeof( int ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProperties.spikeCountOffset, size * sizeof( int ) ) );

    uint64_t* pSpikeHistory[size];
    for (int i = 0; i < size; i++) {
        checkCudaErrors( cudaMalloc( ( void ** ) &pSpikeHistory[i], max_spikes * sizeof( uint64_t ) ) );
    }
    checkCudaErrors( cudaMemcpy ( allNeuronsProperties.spike_history, pSpikeHistory, size * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );
}

/*
 *  Cleanup the class (deallocate memories).
 *
 *  @param  pAllNeuronsProperties_d the AllNeuronsProperties on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllSpikingNeuronsProperties::cleanupNeuronsProperties(void *pAllNeuronsProperties_d, ClusterInfo *clr_info)
{
    AllSpikingNeuronsProperties allNeuronsProperties;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, pAllNeuronsProperties_d, sizeof( AllSpikingNeuronsProperties ), cudaMemcpyDeviceToHost ) );

    deleteNeuronsProperties(allNeuronsProperties);

    checkCudaErrors( cudaFree( pAllNeuronsProperties_d ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProperties   Reference to the AllSpikingNeuronsProperties struct.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllSpikingNeuronsProperties::deleteNeuronsProperties(AllSpikingNeuronsProperties &allNeuronsProperties, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    checkCudaErrors( cudaFree( allNeuronsProperties.hasFired ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.spikeCount ) );
    checkCudaErrors( cudaFree( allNeuronsProperties.spikeCountOffset ) );

    uint64_t* pSpikeHistory[size];
    checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProperties.spike_history, size * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
    for (int i = 0; i < size; i++) {
        checkCudaErrors( cudaFree( pSpikeHistory[i] ) );
    }

    AllNeuronsProperties::deleteNeuronsProperties(allNeuronsProperties);    
}

#endif // USE_GPU
