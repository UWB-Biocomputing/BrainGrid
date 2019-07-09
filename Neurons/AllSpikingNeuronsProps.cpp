#include "AllSpikingNeuronsProps.h"
#include "GPUSpikingCluster.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

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

/*
 *  Clear the spike counts out of all Neurons.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 *  @param  clr       Cluster class to read information from.
 */
void AllSpikingNeuronsProps::clearSpikeCounts(const SimulationInfo *sim_info, const ClusterInfo *clr_info, Cluster *clr)
{
    // clear spike counts in host memory
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    int numNeurons = clr_info->totalClusterNeurons;

    for (int i = 0; i < numNeurons; i++) {
        spikeCountOffset[i] = (spikeCount[i] + spikeCountOffset[i]) % max_spikes;
        spikeCount[i] = 0;
    }

#if defined(USE_GPU)
    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    // clear spike counts in device memory
    AllSpikingNeuronsProps allNeuronsProps;
    AllSpikingNeuronsProps *allNeuronsDeviceProps = dynamic_cast<GPUSpikingCluster*>(clr)->m_allNeuronsDeviceProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllSpikingNeuronsProps ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemset( allNeuronsProps.spikeCount, 0, numNeurons * sizeof( int ) ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.spikeCountOffset, spikeCountOffset, numNeurons * sizeof( int ), cudaMemcpyHostToDevice ) );
#endif // USE_GPU
}

#if defined(USE_GPU)
/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProps   Reference to the AllSpikingNeuronsProps class.
 *  @param  sim_info          SimulationInfo class to read information from.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllSpikingNeuronsProps::allocNeuronsDeviceProps(AllSpikingNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;
    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);

    AllNeuronsProps::allocNeuronsDeviceProps(allNeuronsProps, sim_info, clr_info);

    // allocate GPU memories to store all neuron's states
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.hasFired, size * sizeof( bool ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.spikeCount, size * sizeof( int ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.spikeCountOffset, size * sizeof( int ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.spike_history, size * sizeof( uint64_t* ) ) );

    uint64_t* pSpikeHistory[size];
    for (int i = 0; i < size; i++) {
        checkCudaErrors( cudaMalloc( ( void ** ) &pSpikeHistory[i], max_spikes * sizeof( uint64_t ) ) );
    }
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.spike_history, pSpikeHistory, size * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProps   Reference to the AllSpikingNeuronsProps class.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllSpikingNeuronsProps::deleteNeuronsDeviceProps(AllSpikingNeuronsProps &allNeuronsProps, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    uint64_t* pSpikeHistory[size];
    checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProps.spike_history, size * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
    for (int i = 0; i < size; i++) {
        checkCudaErrors( cudaFree( pSpikeHistory[i] ) );
    }

    checkCudaErrors( cudaFree( allNeuronsProps.hasFired ) );
    checkCudaErrors( cudaFree( allNeuronsProps.spikeCount ) );
    checkCudaErrors( cudaFree( allNeuronsProps.spikeCountOffset ) );
    checkCudaErrors( cudaFree( allNeuronsProps.spike_history ) );

    AllNeuronsProps::deleteNeuronsDeviceProps(allNeuronsProps, clr_info);    
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDeviceProps)
 *
 *  @param  allNeuronsProps    Reference to the AllSpikingNeuronsProps class.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingNeuronsProps::copyHostToDeviceProps( AllSpikingNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    int size = clr_info->totalClusterNeurons;

    checkCudaErrors( cudaMemcpy ( allNeuronsProps.hasFired, hasFired, size * sizeof( bool ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.spikeCount, spikeCount, size * sizeof( int ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allNeuronsProps.spikeCountOffset, spikeCountOffset, size * sizeof( int ), cudaMemcpyHostToDevice ) );

    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
    uint64_t* pSpikeHistory[size];
    checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProps.spike_history, size * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
    for (int i = 0; i < size; i++) {
            checkCudaErrors( cudaMemcpy ( pSpikeHistory[i], spike_history[i], max_spikes * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
    }
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHostProps)
 *
 *  @param  allNeuronsProps    Reference to the AllSpikingNeuronsProps class.
 *  @param  sim_info           SimulationInfo to refer from.
 *  @param  clr_info           ClusterInfo to refer from.
 */
void AllSpikingNeuronsProps::copyDeviceToHostProps( AllSpikingNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info )
{
    int size = clr_info->totalClusterNeurons;

    checkCudaErrors( cudaMemcpy ( hasFired, allNeuronsProps.hasFired, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( spikeCount, allNeuronsProps.spikeCount, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( spikeCountOffset, allNeuronsProps.spikeCountOffset, size * sizeof( int ), cudaMemcpyDeviceToHost ) );

    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
    uint64_t* pSpikeHistory[size];
    checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProps.spike_history, size * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
    for (int i = 0; i < size; i++) {
        checkCudaErrors( cudaMemcpy ( spike_history[i], pSpikeHistory[i], max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
    }
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllSpikingNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllSpikingNeuronsProps::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) 
{
    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    AllSpikingNeuronsProps allNeuronsProps;
    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllSpikingNeuronsProps ), cudaMemcpyDeviceToHost ) );

    int numNeurons = clr_info->totalClusterNeurons;
    uint64_t* pSpikeHistory[numNeurons];
    checkCudaErrors( cudaMemcpy ( pSpikeHistory, allNeuronsProps.spike_history, numNeurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );

    int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
    for (int i = 0; i < numNeurons; i++) {
        checkCudaErrors( cudaMemcpy ( spike_history[i], pSpikeHistory[i],
                max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
    }

    // Set size to 0 to avoid illegal memory deallocation
    // at AllSpikingNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllSpikingNeuronsProps class on device memory.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllSpikingNeuronsProps::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDeviceProps, const ClusterInfo *clr_info )
{
    // Set device ID
    checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

    AllSpikingNeuronsProps allNeuronsProps;
    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, allNeuronsDeviceProps, sizeof( AllSpikingNeuronsProps ), cudaMemcpyDeviceToHost ) );

    int numNeurons = clr_info->totalClusterNeurons;
    checkCudaErrors( cudaMemcpy ( spikeCount, allNeuronsProps.spikeCount, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( spikeCountOffset, allNeuronsProps.spikeCountOffset, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );

    // Set size to 0 to avoid illegal memory deallocation
    // at AllSpikingNeuronsProps deconstructor.
    allNeuronsProps.size = 0;
}

#endif // USE_GPU
