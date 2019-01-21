#include "AllNeuronsProperties.h"

#if !defined(USE_GPU)

// Default constructor
AllNeuronsProperties::AllNeuronsProperties() 
{
    size = 0;
    nParams = 0;
    summation_map = NULL;
}

AllNeuronsProperties::~AllNeuronsProperties()
{
    cleanupNeuronsProperties();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllNeuronsProperties::setupNeuronsProperties(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    size = clr_info->totalClusterNeurons;
    // TODO: Rename variables for easier identification
    summation_map = new BGFLOAT[size];

    for (int i = 0; i < size; ++i) {
        summation_map[i] = 0;
    }

    clr_info->pClusterSummationMap = summation_map;
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllNeuronsProperties::cleanupNeuronsProperties()
{
    if (size != 0) {
        delete[] summation_map;
    }

    summation_map = NULL;
    size = 0;
}

#else // USE_GPU

// Default constructor
AllNeuronsProperties::AllNeuronsProperties() 
{
    summation_map = NULL;
}

AllNeuronsProperties::~AllNeuronsProperties()
{
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  pAllNeuronsProperties_d the AllNeuronsProperties on device memory.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
__host__ static void AllNeuronsProperties::setupNeuronsProperties(void *pAllNeuronsProperties_d, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllNeuronsProperties allNeuronsProperties;

    // allocate GPU memories to store all neuron's states
    allocNeuronsProperties(allNeuronsProperties, sim_info, clr_info);

    // copy the pointer address to structure on device memory
    checkCudaErrors( cudaMemcpy ( pAllNeuronsDeviceProperties_d, &allNeuronsProperties, sizeof( AllNeuronsProperties ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProperties   Reference to the AllNeuronsProperties struct.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllNeuronsProperties::allocNeuronsProperties(AllNeuronsProperties &allNeuronsProperties, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    // allocate GPU memories to store all neuron's states
    checkCudaErrors( cudaMalloc( ( void ** ) allNeuronsProperties.summation_map, size * sizeof( BGFLOAT ) ) );

    // get device summation point address and set it to sim info
    clr_info->pClusterSummationMap = allNeuronsProperties.summation_map;
}

/*
 *  Cleanup the class (deallocate memories).
 *
 *  @param  pAllNeuronsProperties_d the AllNeuronsProperties on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllNeuronsProperties::cleanupNeuronsProperties(void *pAllNeuronsProperties_d, ClusterInfo *clr_info)
{
    AllNeuronsProperties allNeuronsProperties;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProperties, pAllNeuronsProperties_d, sizeof( AllNeuronsProperties ), cudaMemcpyDeviceToHost ) );

    deleteNeuronsProperties(allNeuronsProperties);

    checkCudaErrors( cudaFree( pAllNeuronsProperties_d ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProperties   Reference to the AllNeuronsProperties struct.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllNeuronsProperties::deleteNeuronsProperties(AllNeuronsProperties &allNeuronsProperties, ClusterInfo *clr_info)
{
    checkCudaErrors( cudaFree( allNeuronsProperties.summation_map ) );
}

#endif // USE_GPU

