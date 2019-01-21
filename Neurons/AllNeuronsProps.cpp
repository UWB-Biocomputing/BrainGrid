#include "AllNeuronsProps.h"

#if !defined(USE_GPU)

// Default constructor
AllNeuronsProps::AllNeuronsProps() 
{
    size = 0;
    nParams = 0;
    summation_map = NULL;
}

AllNeuronsProps::~AllNeuronsProps()
{
    cleanupNeuronsProps();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllNeuronsProps::setupNeuronsProps(SimulationInfo *sim_info, ClusterInfo *clr_info)
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
void AllNeuronsProps::cleanupNeuronsProps()
{
    if (size != 0) {
        delete[] summation_map;
    }

    summation_map = NULL;
    size = 0;
}

#else // USE_GPU

// Default constructor
AllNeuronsProps::AllNeuronsProps() 
{
    summation_map = NULL;
}

AllNeuronsProps::~AllNeuronsProps()
{
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  pAllNeuronsProps_d the AllNeuronsProps on device memory.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
__host__ static void AllNeuronsProps::setupNeuronsProps(void *pAllNeuronsProps_d, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllNeuronsProps allNeuronsProps;

    // allocate GPU memories to store all neuron's states
    allocNeuronsProps(allNeuronsProps, sim_info, clr_info);

    // copy the pointer address to structure on device memory
    checkCudaErrors( cudaMemcpy ( pAllNeuronsDeviceProps_d, &allNeuronsProps, sizeof( AllNeuronsProps ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProps   Reference to the AllNeuronsProps struct.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllNeuronsProps::allocNeuronsProps(AllNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    // allocate GPU memories to store all neuron's states
    checkCudaErrors( cudaMalloc( ( void ** ) allNeuronsProps.summation_map, size * sizeof( BGFLOAT ) ) );

    // get device summation point address and set it to sim info
    clr_info->pClusterSummationMap = allNeuronsProps.summation_map;
}

/*
 *  Cleanup the class (deallocate memories).
 *
 *  @param  pAllNeuronsProps_d the AllNeuronsProps on device memory.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllNeuronsProps::cleanupNeuronsProps(void *pAllNeuronsProps_d, ClusterInfo *clr_info)
{
    AllNeuronsProps allNeuronsProps;

    checkCudaErrors( cudaMemcpy ( &allNeuronsProps, pAllNeuronsProps_d, sizeof( AllNeuronsProps ), cudaMemcpyDeviceToHost ) );

    deleteNeuronsProps(allNeuronsProps);

    checkCudaErrors( cudaFree( pAllNeuronsProps_d ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProps   Reference to the AllNeuronsProps struct.
 *  @param  clr_info               ClusterInfo to refer from.
 */
__host__ static void AllNeuronsProps::deleteNeuronsProps(AllNeuronsProps &allNeuronsProps, ClusterInfo *clr_info)
{
    checkCudaErrors( cudaFree( allNeuronsProps.summation_map ) );
}

#endif // USE_GPU

