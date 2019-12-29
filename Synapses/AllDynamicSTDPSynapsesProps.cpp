#include "AllDynamicSTDPSynapsesProps.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

// Default constructor
AllDynamicSTDPSynapsesProps::AllDynamicSTDPSynapsesProps()
{
    lastSpike = NULL;
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
}

AllDynamicSTDPSynapsesProps::~AllDynamicSTDPSynapsesProps()
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
void AllDynamicSTDPSynapsesProps::setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSTDPSynapsesProps::setupSynapsesProps(num_neurons, max_synapses, sim_info, clr_info);

    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        lastSpike = new uint64_t[max_total_synapses];
        r = new BGFLOAT[max_total_synapses];
        u = new BGFLOAT[max_total_synapses];
        D = new BGFLOAT[max_total_synapses];
        U = new BGFLOAT[max_total_synapses];
        F = new BGFLOAT[max_total_synapses];
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllDynamicSTDPSynapsesProps::cleanupSynapsesProps()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] lastSpike;
        delete[] r;
        delete[] u;
        delete[] D;
        delete[] U;
        delete[] F;
    }

    lastSpike = NULL;
    r = NULL;
    u = NULL;
    D = NULL;
    U = NULL;
    F = NULL;
}

#if defined(USE_GPU)
/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllDynamicSTDPSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapsesProps::setupSynapsesDeviceProps( void** allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron )
{
    AllDynamicSTDPSynapsesProps allSynapsesProps;

    allocSynapsesDeviceProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    checkCudaErrors( cudaMalloc( allSynapsesDeviceProps, sizeof( AllDynamicSTDPSynapsesProps ) ) );
    checkCudaErrors( cudaMemcpy ( *allSynapsesDeviceProps, &allSynapsesProps, sizeof( AllDynamicSTDPSynapsesProps ), cudaMemcpyHostToDevice ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;
}

/*
 *  Allocate GPU memories to store all synapses' states.
 *
 *  @param  allSynapsesProps      Reference to the AllDynamicSTDPSynapsesProps class.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapsesProps::allocSynapsesDeviceProps( AllDynamicSTDPSynapsesProps &allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSTDPSynapsesProps::allocSynapsesDeviceProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron);

    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.lastSpike, size * sizeof( uint64_t ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.r, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.u, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.D, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.U, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.F, size * sizeof( BGFLOAT ) ) );
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDeviceProps  Reference to the AllDynamicSTDPSynapsesProps class on device memory.
 */
void AllDynamicSTDPSynapsesProps::cleanupSynapsesDeviceProps( void* allSynapsesDeviceProps )
{
    AllDynamicSTDPSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDynamicSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    deleteSynapsesDeviceProps( allSynapsesProps );

    checkCudaErrors( cudaFree( allSynapsesDeviceProps ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllDynamicSTDPSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesProps  Reference to the AllDynamicSTDPSynapsesProps class.
 */
void AllDynamicSTDPSynapsesProps::deleteSynapsesDeviceProps( AllDynamicSTDPSynapsesProps& allSynapsesProps )
{
    checkCudaErrors( cudaFree( allSynapsesProps.lastSpike ) );
    checkCudaErrors( cudaFree( allSynapsesProps.r ) );
    checkCudaErrors( cudaFree( allSynapsesProps.u ) );
    checkCudaErrors( cudaFree( allSynapsesProps.D ) );
    checkCudaErrors( cudaFree( allSynapsesProps.U ) );
    checkCudaErrors( cudaFree( allSynapsesProps.F ) );

    AllSTDPSynapsesProps::deleteSynapsesDeviceProps( allSynapsesProps );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllDynamicSTDPSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapsesProps::copySynapseHostToDeviceProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron )
{
    AllDynamicSTDPSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDynamicSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    copyHostToDeviceProps( allSynapsesDeviceProps, allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllDynamicSTDPSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDeviceProps)
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllDynamicSTDPSynapsesProps class on device memory.
 *  @param  allSynapsesProps         Reference to the AllDynamicSTDPSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapsesProps::copyHostToDeviceProps( void* allSynapsesDeviceProps, AllDynamicSTDPSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron )
{
    // copy everything necessary
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSTDPSynapsesProps::copyHostToDeviceProps( allSynapsesDeviceProps, allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    checkCudaErrors( cudaMemcpy ( allSynapsesProps.lastSpike, lastSpike,
            size * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.r, r,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.u, u,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.D, D,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.U, U,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.F, F,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllDynamicSTDPSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapsesProps::copySynapseDeviceToHostProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron )
{
    AllDynamicSTDPSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDynamicSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    copyDeviceToHostProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllDynamicSTDPSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;

}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHostProps)
 *
 *  @param  allSynapsesProps         Reference to the AllDynamicSTDPSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllDynamicSTDPSynapsesProps::copyDeviceToHostProps( AllDynamicSTDPSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSTDPSynapsesProps::copyDeviceToHostProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron);

    checkCudaErrors( cudaMemcpy ( lastSpike, allSynapsesProps.lastSpike,
            size * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( r, allSynapsesProps.r,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( u, allSynapsesProps.u,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( D, allSynapsesProps.D,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( U, allSynapsesProps.U,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( F, allSynapsesProps.F,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}
#endif // USE_GPU

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllDynamicSTDPSynapsesProps::readSynapseProps(istream &input, const BGSIZE iSyn)
{
    AllSTDPSynapsesProps::readSynapseProps(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> lastSpike[iSyn]; input.ignore();
    input >> r[iSyn]; input.ignore();
    input >> u[iSyn]; input.ignore();
    input >> D[iSyn]; input.ignore();
    input >> U[iSyn]; input.ignore();
    input >> F[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllDynamicSTDPSynapsesProps::writeSynapseProps(ostream& output, const BGSIZE iSyn) const
{
    AllSTDPSynapsesProps::writeSynapseProps(output, iSyn);

    output << lastSpike[iSyn] << ends;
    output << r[iSyn] << ends;
    output << u[iSyn] << ends;
    output << D[iSyn] << ends;
    output << U[iSyn] << ends;
    output << F[iSyn] << ends;
}

/*
 *  Prints all SynapsesProps data.
 */
void AllDynamicSTDPSynapsesProps::printSynapsesProps() 
{
    AllSTDPSynapsesProps::printSynapsesProps();
    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        if (W[i] != 0.0) {
            cout << "lastSpike: " << lastSpike[i];
            cout << " r: " << r[i];
            cout << " u: " << u[i];
            cout << " D: " << D[i];
            cout << " U: " << U[i];
            cout << " F: " << F[i] << endl;
        } else {
            if(lastSpike[i] != 0.0 || r[i] != 0.0 || u[i] != 0.0 || D[i] != 0.0 || U[i] != 0.0 || F[i] != 0.0) {
                cout << "---------------------ERROR!!!!!!!!-------------";
                cout << " lastSpike: " << lastSpike[i];
                cout << " r: " << r[i];
                cout << " u: " << u[i];
                cout << " D: " << D[i];
                cout << " U: " << U[i];
                cout << " F: " << F[i] << endl;
            }
        }
    }
}
