#include "AllSpikingNeuronsProps.h"

#if !defined(USE_GPU)

// Default constructor
AllSpikingNeuronsProps::AllSpikingNeuronsProps()
{
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

AllSpikingNeuronsProps::~AllSpikingNeuronsProps()
{
    cleanupNeuronsProps();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingNeuronsProps::setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllNeuronsProps::setupNeuronsProps(sim_info, clr_info);

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
void AllSpikingNeuronsProps::cleanupNeuronsProps()
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
AllSpikingNeuronsProps::AllSpikingNeuronsProps()
{
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

AllSpikingNeuronsProps::~AllSpikingNeuronsProps()
{
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  pAllNeuronsProps_d the AllNeuronsProps on device memory.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
__host__ static void AllSpikingNeuronsProps::setupNeuronsProps(void *pAllNeuronsProps_d, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingNeuronsProps allNeuronsProps;

    // allocate GPU memories to store all neuron's states
    allocNeuronsProps(allNeuronsProps, sim_info, clr_info);

    // copy the pointer address to structure on device memory
    checkCudaErrors( cudaMemcpy ( pAllNeuronsDeviceProps_d, &allNeuronsProps, sizeof( AllSpikingNeuronsProps ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProps   Reference to the AllSpikingNeuronsProps struct.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllSpikingNeuronsProps::allocNeuronsProps(AllSpikingNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);

    AllNeuronsProps::allocNeuronsProps(allNeuronsProps, clr_info);

    // allocate GPU memories to store all neuron's states
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.hasFired, size * sizeof( bool ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.spikeCount, size * sizeof( int ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.spikeCountOffset, size * sizeof( int ) ) );

    uint64_t* pSpikeHistory[size];
    for (int i = 0; i < size; i++) {
        checkCudaErrors( cudaMalloc( ( void ** ) &pSpikeHistory[i], max_spikes * sizeof( uint64_t ) ) );
    }
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.spike_history, pSpikeHistory, size * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );
}

/*
 *  Cleanup the class (deallocate memories).
 *
 *  @param  pAllNeuronsProps_d the AllNeuronsProps on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllSpikingNeuronsProps::cleanupNeuronsProps(void *pAllNeuronsProps_d, ClusterInfo *clr_info)
{
    AllSpikingNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, pAllNeuronsProps_d, sizeof( AllSpikingNeuronsProps ), cudaMemcpyDeviceToHost ) );

    deleteNeuronsProps(allNeuronsProps);

    checkCudaErrors( cudaFree( pAllNeuronsProps_d ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProps   Reference to the AllSpikingNeuronsProps struct.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllSpikingNeuronsProps::deleteNeuronsProps(AllSpikingNeuronsProps &allNeuronsProps, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    checkCudaErrors( cudaFree( allNeuronsProps.hasFired ) );
    checkCudaErrors( cudaFree( allNeuronsProps.spikeCount ) );
    checkCudaErrors( cudaFree( allNeuronsProps.spikeCountOffset ) );

    uint64_t* pSpikeHistory[size];
    checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProps.spike_history, size * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
    for (int i = 0; i < size; i++) {
        checkCudaErrors( cudaFree( pSpikeHistory[i] ) );
    }

    AllNeuronsProps::deleteNeuronsProps(allNeuronsProps);    
}

#endif // USE_GPU
