#include "AllNeuronsProps.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

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

#if defined(USE_GPU)
/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIZHNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllNeuronsProps::setupNeuronsDeviceProps(void** allNeuronsDeviceProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllNeuronsProps::cleanupNeuronsDeviceProps(void *allNeuronsDeviceProps, ClusterInfo *clr_info)
{
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllNeuronsProps::copyNeuronHostToDeviceProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info )
{
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDeviceProps   Reference to the AllIFNeuronsProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllNeuronsProps::copyNeuronDeviceToHostProps( void* allNeuronsDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info )
{
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *
 *  @param  allNeuronsProps   Reference to the AllNeuronsProps class.
 *  @param  sim_info          SimulationInfo class to read information from.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllNeuronsProps::allocNeuronsDeviceProps(AllNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    int size = clr_info->totalClusterNeurons;

    // allocate GPU memories to store all neuron's states
    checkCudaErrors( cudaMalloc( ( void ** ) &allNeuronsProps.summation_map, size * sizeof( BGFLOAT ) ) );

    // get device summation point address and set it to sim info
    clr_info->pClusterSummationMap = allNeuronsProps.summation_map;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsProps   Reference to the AllNeuronsProps class.
 *  @param  clr_info          ClusterInfo to refer from.
 */
void AllNeuronsProps::deleteNeuronsDeviceProps(AllNeuronsProps &allNeuronsProps, ClusterInfo *clr_info)
{
    checkCudaErrors( cudaFree( allNeuronsProps.summation_map ) );
}
#endif // USE_GPU

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllNeuronsProps::checkNumParameters()
{
    return true;
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllNeuronsProps::readParameters(const TiXmlElement& element)
{
    return true;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllNeuronsProps::printParameters(ostream &output) const
{
}

/*
 *  Copy neurons parameters.
 *
 *  @param  r_neurons  Neurons properties class object to copy from.
 */
void AllNeuronsProps::copyParameters(const AllNeuronsProps *r_neuronsProps)
{
}

/*
 *  Sets the data for Neuron #index to input's data.
 *
 *  @param  input       istream to read from.
 *  @param  i           index of the neuron (in neurons).
 */
void AllNeuronsProps::readNeuronProps(istream &input, int i)
{
}

/*
 *  Writes out the data in the selected Neuron.
 *
 *  @param  output      stream to write out to.
 *  @param  i           index of the neuron (in neurons).
 */
void AllNeuronsProps::writeNeuronProps(ostream& output, int i) const
{
}

/*
 *  Creates a single Neuron and generates data for it.
 *
 *  @param  sim_info     SimulationInfo class to read information from.
 *  @param  neuron_index Index of the neuron to create.
 *  @param  layout       Layout information of the neunal network.
 *  @param  clr_info     ClusterInfo class to read information from.
 */
void AllNeuronsProps::setNeuronPropValues(SimulationInfo *sim_info, int neuron_index, Layout *layout, ClusterInfo *clr_info)
{
}

/*
 *  Set the Neuron at the indexed location to default values.
 *
 *Inoise  @param  neuron_index    Index of the Neuron that the synapse belongs to.
 */
void AllNeuronsProps::setNeuronPropDefaults(const int index)
{
}

