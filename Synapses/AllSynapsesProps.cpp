#include "AllSynapsesProps.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

// Default constructor
AllSynapsesProps::AllSynapsesProps()
{
    nParams = 0;

    destNeuronLayoutIndex = nullptr;
    W = nullptr;
    summationPoint = nullptr;
    sourceNeuronLayoutIndex = nullptr;
    psr = nullptr;
    type = nullptr;
    in_use = nullptr;
    synapse_counts = nullptr;
    maxSynapsesPerNeuron = 0;
    count_neurons = 0;
}

AllSynapsesProps::~AllSynapsesProps()
{
    cleanupSynapsesProps();
}

/*
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *
 *  @param  num_neurons   Total number of neurons in the network.
 *  @param  max_synapses  Maximum number of synapses per neuron.
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSynapsesProps::setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    count_neurons = num_neurons;
    maxSynapsesPerNeuron = max_synapses;
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    total_synapse_counts = 0;

    if (max_total_synapses != 0) {
        destNeuronLayoutIndex = new int[max_total_synapses];
        W = new BGFLOAT[max_total_synapses];
        summationPoint = new BGFLOAT*[max_total_synapses];
        sourceNeuronLayoutIndex = new int[max_total_synapses];
        psr = new BGFLOAT[max_total_synapses];
        type = new synapseType[max_total_synapses];
        in_use = new bool[max_total_synapses];
        synapse_counts = new BGSIZE[num_neurons];

        for (BGSIZE i = 0; i < max_total_synapses; i++) {
            summationPoint[i] = nullptr;
            in_use[i] = false;
            W[i] = 0;
        }

        for (int i = 0; i < num_neurons; i++) {
            synapse_counts[i] = 0;
        }
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllSynapsesProps::cleanupSynapsesProps()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] destNeuronLayoutIndex;
        delete[] W;
        delete[] summationPoint;
        delete[] sourceNeuronLayoutIndex;
        delete[] psr;
        delete[] type;
        delete[] in_use;
        delete[] synapse_counts;
    }

    destNeuronLayoutIndex = nullptr;
    W = nullptr;
    summationPoint = nullptr;
    sourceNeuronLayoutIndex = nullptr;
    psr = nullptr;
    type = nullptr;
    in_use = nullptr;
    synapse_counts = nullptr;

    count_neurons = 0;
    maxSynapsesPerNeuron = 0;
}

#if defined(USE_GPU)
/*
 *  Allocate GPU memories to store all synapses' states.
 *
 *  @param  allSynapsesProps      Reference to the AllSynapsesProps class.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSynapsesProps::allocSynapsesDeviceProps( AllSynapsesProps &allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.sourceNeuronLayoutIndex, size * sizeof( int ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.destNeuronLayoutIndex, size * sizeof( int ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.W, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.summationPoint, size * sizeof( BGFLOAT* ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.type, size * sizeof( synapseType ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.psr, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.in_use, size * sizeof( bool ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.synapse_counts, num_neurons * sizeof( BGSIZE ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.summation, size * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesProps  Reference to the AllSynapsesProps class.
 */
void AllSynapsesProps::deleteSynapsesDeviceProps( AllSynapsesProps& allSynapsesProps ) 
{
    checkCudaErrors( cudaFree( allSynapsesProps.sourceNeuronLayoutIndex ) );
    checkCudaErrors( cudaFree( allSynapsesProps.destNeuronLayoutIndex ) );
    checkCudaErrors( cudaFree( allSynapsesProps.W ) );
    checkCudaErrors( cudaFree( allSynapsesProps.summationPoint ) );
    checkCudaErrors( cudaFree( allSynapsesProps.type ) );
    checkCudaErrors( cudaFree( allSynapsesProps.psr ) );
    checkCudaErrors( cudaFree( allSynapsesProps.in_use ) );
    checkCudaErrors( cudaFree( allSynapsesProps.synapse_counts ) );
    checkCudaErrors( cudaFree( allSynapsesProps.summation ) );
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDeviceProps)
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSynapsesProps class on device memory.
 *  @param  allSynapsesProps         Reference to the AllSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSynapsesProps::copyHostToDeviceProps( void* allSynapsesDeviceProps, AllSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron ) 
{ 
    // copy everything necessary
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    allSynapsesProps.maxSynapsesPerNeuron = maxSynapsesPerNeuron;
    allSynapsesProps.total_synapse_counts = total_synapse_counts;
    allSynapsesProps.count_neurons = count_neurons;
    checkCudaErrors( cudaMemcpy ( allSynapsesDeviceProps, &allSynapsesProps, sizeof( AllSynapsesProps ), cudaMemcpyHostToDevice ) );

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;

    checkCudaErrors( cudaMemcpy ( allSynapsesProps.sourceNeuronLayoutIndex, sourceNeuronLayoutIndex,
            size * sizeof( int ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.destNeuronLayoutIndex, destNeuronLayoutIndex,
            size * sizeof( int ),  cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.W, W,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.type, type,
            size * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.psr, psr,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.in_use, in_use,
            size * sizeof( bool ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.synapse_counts, synapse_counts,
                    num_neurons * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHostProps)
 *
 *  @param  allSynapsesProps         Reference to the AllSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSynapsesProps::copyDeviceToHostProps( AllSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapsesProps.synapse_counts,
            num_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
    this->maxSynapsesPerNeuron = allSynapsesProps.maxSynapsesPerNeuron;
    this->total_synapse_counts = allSynapsesProps.total_synapse_counts;
    this->count_neurons = allSynapsesProps.count_neurons;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;

    checkCudaErrors( cudaMemcpy ( sourceNeuronLayoutIndex, allSynapsesProps.sourceNeuronLayoutIndex,
            size * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( destNeuronLayoutIndex, allSynapsesProps.destNeuronLayoutIndex,
            size * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( W, allSynapsesProps.W,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( type, allSynapsesProps.type,
            size * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( psr, allSynapsesProps.psr,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( in_use, allSynapsesProps.in_use,
            size * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}
#endif // USE_GPU

/*
 *  Checks the number of required parameters.
 *
 * @return true if all required parameters were successfully read, false otherwise.
 */
bool AllSynapsesProps::checkNumParameters()
{
    return (nParams >= 0);
}

/*
 *  Attempts to read parameters from a XML file.
 *
 *  @param  element TiXmlElement to examine.
 *  @return true if successful, false otherwise.
 */
bool AllSynapsesProps::readParameters(const TiXmlElement& element)
{
    return false;
}

/*
 *  Prints out all parameters of the neurons to ostream.
 *
 *  @param  output  ostream to send output to.
 */
void AllSynapsesProps::printParameters(ostream &output) const
{
}

/*
 *  Copy synapses parameters.
 *
 *  @param  r_synapsesProps  Synapses properties class object to copy from.
 */
void AllSynapsesProps::copyParameters(const AllSynapsesProps *r_synapsesProps)
{
}

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSynapsesProps::readSynapseProps(istream &input, const BGSIZE iSyn)
{
    int synapse_type(0);

    // input.ignore() so input skips over end-of-line characters.
    input >> sourceNeuronLayoutIndex[iSyn]; input.ignore();
    input >> destNeuronLayoutIndex[iSyn]; input.ignore();
    input >> W[iSyn]; input.ignore();
    input >> psr[iSyn]; input.ignore();
    input >> synapse_type; input.ignore();
    input >> in_use[iSyn]; input.ignore();

    type[iSyn] = synapseOrdinalToType(synapse_type);
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSynapsesProps::writeSynapseProps(ostream& output, const BGSIZE iSyn) const
{
    output << sourceNeuronLayoutIndex[iSyn] << ends;
    output << destNeuronLayoutIndex[iSyn] << ends;
    output << W[iSyn] << ends;
    output << psr[iSyn] << ends;
    output << type[iSyn] << ends;
    output << in_use[iSyn] << ends;
}

/*
 *  Returns an appropriate synapseType object for the given integer.
 *
 *  @param  type_ordinal    Integer that correspond with a synapseType.
 *  @return the SynapseType that corresponds with the given integer.
 */
synapseType AllSynapsesProps::synapseOrdinalToType(const int type_ordinal)
{
        switch (type_ordinal) {
        case 0:
                return II;
        case 1:
                return IE;
        case 2:
                return EI;
        case 3:
                return EE;
        default:
                return STYPE_UNDEF;
        }
}

/*
 *  Prints SynapsesProps data.
 */
void AllSynapsesProps::printSynapsesProps() const
{
    cout << "This is SynapsesProps data:" << endl;
    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        if (W[i] != 0.0) {
                cout << "W[" << i << "] = " << W[i];
                cout << " sourNeuron: " << sourceNeuronLayoutIndex[i];
                cout << " desNeuron: " << destNeuronLayoutIndex[i];
                cout << " type: " << type[i];
                cout << " psr: " << psr[i];
                cout << " in_use:" << in_use[i];
                if(summationPoint[i] != nullptr) {
                     cout << " summationPoint: is created!" << endl;    
                } else {
                     cout << " summationPoint: is EMPTY!!!!!" << endl;  
                }
        }
    }
    
    for (int i = 0; i < count_neurons; i++) {
        cout << "synapse_counts:" << "neuron[" << i  << "]" << synapse_counts[i] << endl;
    }
    
    cout << "total_synapse_counts:" << total_synapse_counts << endl;
    cout << "maxSynapsesPerNeuron:" << maxSynapsesPerNeuron << endl;
    cout << "count_neurons:" << count_neurons << endl;
}
