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
        }
    }
}

#if defined(USE_GPU)
/**
 *  Prints all GPU SynapsesProps data.
 * 
 *  @param  allSynapsesDeviceProps   Reference to the AllSTDPSynapsesProps class on device memory.
 */
void AllDynamicSTDPSynapsesProps::printGPUSynapsesProps( void* allSynapsesDeviceProps ) 
{
    AllDynamicSTDPSynapsesProps allSynapsesProps;

    //allocate print out data members
    BGSIZE size = maxSynapsesPerNeuron * count_neurons;

    BGSIZE *synapse_countsPrint = new BGSIZE[count_neurons];
    BGSIZE maxSynapsesPerNeuronPrint;
    BGSIZE total_synapse_countsPrint;
    int count_neuronsPrint;
    int *sourceNeuronLayoutIndexPrint = new int[size];
    int *destNeuronLayoutIndexPrint = new int[size];
    BGFLOAT *WPrint = new BGFLOAT[size];

    synapseType *typePrint = new synapseType[size];
    BGFLOAT *psrPrint = new BGFLOAT[size];
    bool *in_usePrint = new bool[size];

    for (BGSIZE i = 0; i < size; i++) {
        in_usePrint[i] = false;
    }

    for (int i = 0; i < count_neurons; i++) {
        synapse_countsPrint[i] = 0;
    }

    BGFLOAT *decayPrint = new BGFLOAT[size];
    int *total_delayPrint = new int[size];
    BGFLOAT *tauPrint = new BGFLOAT[size];

    int *total_delayPostPrint = new int[size];
    BGFLOAT *tauspostPrint = new BGFLOAT[size];
    BGFLOAT *tausprePrint = new BGFLOAT[size];
    BGFLOAT *tauposPrint = new BGFLOAT[size];
    BGFLOAT *taunegPrint = new BGFLOAT[size];
    BGFLOAT *STDPgapPrint = new BGFLOAT[size];
    BGFLOAT *WexPrint = new BGFLOAT[size];
    BGFLOAT *AnegPrint = new BGFLOAT[size];
    BGFLOAT *AposPrint = new BGFLOAT[size];
    BGFLOAT *muposPrint = new BGFLOAT[size];
    BGFLOAT *munegPrint = new BGFLOAT[size];
    bool *useFroemkeDanSTDPPrint = new bool[size];

    uint64_t *lastSpikePrint = new uint64_t[size];
    BGFLOAT *rPrint = new BGFLOAT[size];
    BGFLOAT *uPrint = new BGFLOAT[size];
    BGFLOAT *DPrint = new BGFLOAT[size];
    BGFLOAT *UPrint = new BGFLOAT[size];
    BGFLOAT *FPrint = new BGFLOAT[size];
    
    // copy everything
    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllDynamicSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( synapse_countsPrint, allSynapsesProps.synapse_counts, count_neurons * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );
    maxSynapsesPerNeuronPrint = allSynapsesProps.maxSynapsesPerNeuron;
    total_synapse_countsPrint = allSynapsesProps.total_synapse_counts;
    count_neuronsPrint = allSynapsesProps.count_neurons;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;

    checkCudaErrors( cudaMemcpy ( sourceNeuronLayoutIndexPrint, allSynapsesProps.sourceNeuronLayoutIndex, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( destNeuronLayoutIndexPrint, allSynapsesProps.destNeuronLayoutIndex, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( WPrint, allSynapsesProps.W, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( typePrint, allSynapsesProps.type, size * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( psrPrint, allSynapsesProps.psr, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( in_usePrint, allSynapsesProps.in_use, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaMemcpy ( decayPrint, allSynapsesProps.decay, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( tauPrint, allSynapsesProps.tau, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( total_delayPrint, allSynapsesProps.total_delay,size * sizeof( int ), cudaMemcpyDeviceToHost ) );
    
    checkCudaErrors( cudaMemcpy ( total_delayPostPrint, allSynapsesProps.total_delayPost, size * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( tauspostPrint, allSynapsesProps.tauspost, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( tausprePrint, allSynapsesProps.tauspre, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( tauposPrint, allSynapsesProps.taupos, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( taunegPrint, allSynapsesProps.tauneg, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( STDPgapPrint, allSynapsesProps.STDPgap, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( WexPrint, allSynapsesProps.Wex, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( AnegPrint, allSynapsesProps.Aneg, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( AposPrint, allSynapsesProps.Apos, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( muposPrint, allSynapsesProps.mupos, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( munegPrint, allSynapsesProps.muneg, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( useFroemkeDanSTDPPrint, allSynapsesProps.useFroemkeDanSTDP, size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

    checkCudaErrors( cudaMemcpy ( lastSpikePrint, allSynapsesProps.lastSpike, size * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( rPrint, allSynapsesProps.r, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( uPrint, allSynapsesProps.u, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( DPrint, allSynapsesProps.D, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( UPrint, allSynapsesProps.U, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( FPrint, allSynapsesProps.F, size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );

    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        if (WPrint[i] != 0.0) {
            cout << "W[" << i << "] = " << WPrint[i];
            cout << " sourNeuron: " << sourceNeuronLayoutIndexPrint[i];
            cout << " desNeuron: " << destNeuronLayoutIndexPrint[i];
            cout << " type: " << typePrint[i];
            cout << " psr: " << psrPrint[i];
            cout << " in_use:" << in_usePrint[i];

            cout << "decay: " << decayPrint[i];
            cout << " tau: " << tauPrint[i];
            cout << " total_delay: " << total_delayPrint[i];

            cout << "total_delayPost: " << total_delayPostPrint[i];
            cout << " tauspost: " << tauspostPrint[i];
            cout << " tauspre: " << tausprePrint[i];
            cout << " taupos: " << tauposPrint[i];
            cout << " tauneg: " << taunegPrint[i];
            cout << " STDPgap: " << STDPgapPrint[i];
            cout << " Wex: " << WexPrint[i];
            cout << " Aneg: " << AnegPrint[i];
            cout << " Apos: " << AposPrint[i];
            cout << " mupos: " << muposPrint[i];
            cout << " muneg: " << munegPrint[i];
            cout << " useFroemkeDanSTDP: " << useFroemkeDanSTDPPrint[i];

            cout << "lastSpike: " << lastSpikePrint[i];
            cout << " r: " << rPrint[i];
            cout << " u: " << uPrint[i];
            cout << " D: " << DPrint[i];
            cout << " U: " << UPrint[i];
            cout << " F: " << FPrint[i] << endl;
        }
    }

    for (int i = 0; i < count_neurons; i++) {
        cout << "synapse_counts:" << "[" << i  << "]" << synapse_countsPrint[i] << " ";
    }
    cout << endl;
    
    cout << "GPU total_synapse_counts:" << total_synapse_countsPrint << endl;
    cout << "GPU maxSynapsesPerNeuron:" << maxSynapsesPerNeuronPrint << endl;
    cout << "GPU count_neurons:" << count_neuronsPrint << endl;

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllDSSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}
#endif // USE_GPU
