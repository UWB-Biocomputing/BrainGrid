#include "AllSpikingSynapsesProps.h"
#include "EventQueue.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

// Default constructor
AllSpikingSynapsesProps::AllSpikingSynapsesProps()
{
    decay = NULL;
    total_delay = NULL;
    tau = NULL;
    preSpikeQueue = NULL;
}

AllSpikingSynapsesProps::~AllSpikingSynapsesProps()
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
void AllSpikingSynapsesProps::setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSynapsesProps::setupSynapsesProps(num_neurons, max_synapses, sim_info, clr_info);

    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        decay = new BGFLOAT[max_total_synapses];
        total_delay = new int[max_total_synapses];
        tau = new BGFLOAT[max_total_synapses];

        // create a pre synapse spike queue & initialize it
        preSpikeQueue = new EventQueue();
#if defined(USE_GPU)
        // max_total_synapses * sim_info->maxFiringRate may overflow the maximum range
        // of 32 bits integer, so we cast it uint64_t
        int nMaxInterClustersOutgoingEvents = (uint64_t) max_total_synapses * sim_info->maxFiringRate * sim_info->deltaT * sim_info->minSynapticTransDelay;
        int nMaxInterClustersIncomingEvents = (uint64_t) max_total_synapses * sim_info->maxFiringRate * sim_info->deltaT * sim_info->minSynapticTransDelay;

        // initializes the pre synapse spike queue
        preSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses, nMaxInterClustersOutgoingEvents, nMaxInterClustersIncomingEvents);
#else // USE_GPU
        // initializes the pre synapse spike queue
        preSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses);
#endif // USE_GPU

        // register the queue to the event handler
        if (clr_info->eventHandler != NULL) {
            clr_info->eventHandler->addEventQueue(clr_info->clusterID, preSpikeQueue);
        }
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllSpikingSynapsesProps::cleanupSynapsesProps()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] decay;
        delete[] total_delay;
        delete[] tau;
    }

    decay = NULL;
    total_delay = NULL;
    tau = NULL;

    if (preSpikeQueue != NULL) {
        delete preSpikeQueue;
        preSpikeQueue = NULL;
    }
}

#if defined(USE_GPU)
/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSpikingSynapsesProps::setupSynapsesDeviceProps( void** allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron ) 
{
    AllSpikingSynapsesProps allSynapsesProps;
    allocSynapsesDeviceProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    checkCudaErrors( cudaMalloc( allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ) ) );
    checkCudaErrors( cudaMemcpy ( *allSynapsesDeviceProps, &allSynapsesProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyHostToDevice ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;
}

/*
 *  Allocate GPU memories to store all synapses' states.
 *
 *  @param  allSynapsesProps      Reference to the AllSpikingSynapsesProps class.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSpikingSynapsesProps::allocSynapsesDeviceProps( AllSpikingSynapsesProps &allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSynapsesProps::allocSynapsesDeviceProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron);

    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.decay, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.tau, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.total_delay, size * sizeof( int ) ) );

    // create an EventQueue objet in device memory and set the pointer in device
    preSpikeQueue->createEventQueueInDevice(&allSynapsesProps.preSpikeQueue);
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDeviceProps  Reference to the AllSpikingSynapsesProps class on device memory.
 */
void AllSpikingSynapsesProps::cleanupSynapsesDeviceProps( void* allSynapsesDeviceProps ) 
{
    AllSpikingSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );
    deleteSynapsesDeviceProps( allSynapsesProps );

    checkCudaErrors( cudaFree( allSynapsesDeviceProps ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSpikingSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesProps  Reference to the AllSpikingSynapsesProps class.
 */
void AllSpikingSynapsesProps::deleteSynapsesDeviceProps( AllSpikingSynapsesProps& allSynapsesProps )
{
    checkCudaErrors( cudaFree( allSynapsesProps.decay ) );
    checkCudaErrors( cudaFree( allSynapsesProps.tau ) );
    checkCudaErrors( cudaFree( allSynapsesProps.total_delay ) );

    // delete EventQueue object in device memory.
    EventQueue::deleteEventQueueInDevice(allSynapsesProps.preSpikeQueue);

    AllSynapsesProps::deleteSynapsesDeviceProps( allSynapsesProps );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSpikingSynapsesProps::copySynapseHostToDeviceProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron ) 
{
    AllSpikingSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );
    copyHostToDeviceProps( allSynapsesDeviceProps, allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSpikingSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDeviceProps)
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
 *  @param  allSynapsesProps         Reference to the AllSpikingSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSpikingSynapsesProps::copyHostToDeviceProps( void* allSynapsesDeviceProps, AllSpikingSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron )
{
    // copy everything necessary
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSynapsesProps::copyHostToDeviceProps( allSynapsesDeviceProps, allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    checkCudaErrors( cudaMemcpy ( allSynapsesProps.decay, decay,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.tau, tau,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.total_delay, total_delay,
            size * sizeof( int ), cudaMemcpyHostToDevice ) );

    // copy event queue data from host to device.
    preSpikeQueue->copyEventQueueHostToDevice(allSynapsesProps.preSpikeQueue);
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSpikingSynapsesProps::copySynapseDeviceToHostProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron )
{
    AllSpikingSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );
    copyDeviceToHostProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSpikingSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHostProps)
 *
 *  @param  allSynapsesProps         Reference to the AllSpikingSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSpikingSynapsesProps::copyDeviceToHostProps( AllSpikingSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSynapsesProps::copyDeviceToHostProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron);

    checkCudaErrors( cudaMemcpy ( decay, allSynapsesProps.decay,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( tau, allSynapsesProps.tau,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( total_delay, allSynapsesProps.total_delay,
            size * sizeof( int ), cudaMemcpyDeviceToHost ) );

    // copy event queue data from device to host.
    preSpikeQueue->copyEventQueueDeviceToHost(allSynapsesProps.preSpikeQueue);
}

/*
 *  Get synapse_counts in AllSpikingSynapsesProps class on device memory.
 *
 *  @param  allSynapsesDeviceProps  Reference to the AllSpikingSynapsesProps class on device memory.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllSpikingSynapsesProps::copyDeviceSynapseCountsToHost(void* allSynapsesDeviceProps, const ClusterInfo *clr_info)
{
    AllSpikingSynapsesProps allSynapsesProps;
    int neuron_count = clr_info->totalClusterNeurons;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( synapse_counts, allSynapsesProps.synapse_counts, neuron_count * sizeof( BGSIZE ), cudaMemcpyDeviceToHost ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSpikingSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Get sourceNeuronLayoutIndex and in_use in AllSpikingSynapsesProps class on device memory.
 *
 *  @param  allSynapsesDeviceProps  Reference to the AllSpikingSynapsesProps class on device memory.
 *  @param  sim_info                SimulationInfo to refer from.
 *  @param  clr_info                ClusterInfo to refer from.
 */
void AllSpikingSynapsesProps::copyDeviceSourceNeuronIdxToHost(void* allSynapsesDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
    AllSpikingSynapsesProps allSynapsesProps;
    BGSIZE max_total_synapses = sim_info->maxSynapsesPerNeuron * clr_info->totalClusterNeurons;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( sourceNeuronLayoutIndex, allSynapsesProps.sourceNeuronLayoutIndex,
            max_total_synapses * sizeof( int ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( in_use, allSynapsesProps.in_use,
            max_total_synapses * sizeof( bool ), cudaMemcpyDeviceToHost ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSpikingSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 * Process inter clusters outgoing spikes.
 *
 *  @param  allSynapsesDeviceProps     Reference to the AllSpikingSynapsesProps struct
 *                                on device memory.
 */
void AllSpikingSynapsesProps::processInterClustesOutgoingSpikes(void* allSynapsesDeviceProps)
{
    // copy everything necessary from device to host
    AllSpikingSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );

    // process inter clusters outgoing spikes
    preSpikeQueue->processInterClustersOutgoingEvents(allSynapsesProps.preSpikeQueue);

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSpikingSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 * Process inter clusters incoming spikes.
 *
 *  @param  allSynapsesDeviceProps     Reference to the AllSpikingSynapsesProps struct
 *                                on device memory.
 */
void AllSpikingSynapsesProps::processInterClustesIncomingSpikes(void* allSynapsesDeviceProps)
{
    // copy everything necessary from host to device
    AllSpikingSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );

    // process inter clusters incoming spikes
    preSpikeQueue->processInterClustersIncomingEvents(allSynapsesProps.preSpikeQueue);

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSpikingSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}
#endif // USE_GPU

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSpikingSynapsesProps::readSynapseProps(istream &input, const BGSIZE iSyn)
{
    AllSynapsesProps::readSynapseProps(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> decay[iSyn]; input.ignore();
    input >> total_delay[iSyn]; input.ignore();
    input >> tau[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSpikingSynapsesProps::writeSynapseProps(ostream& output, const BGSIZE iSyn) const
{
    AllSynapsesProps::writeSynapseProps(output, iSyn);

    output << decay[iSyn] << ends;
    output << total_delay[iSyn] << ends;
    output << tau[iSyn] << ends;
}


/*
 *  Prints all SynapsesProps data.
 */
void AllSpikingSynapsesProps::printSynapsesProps() 
{
    AllSynapsesProps::printSynapsesProps();
    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        if (W[i] != 0.0) {
            cout << "decay: " << decay[i];
            cout << " tau: " << tau[i];
            cout << " total_delay: " << total_delay[i];
            cout << " preSpikeQueue: " << preSpikeQueue->m_queueEvent[i] << endl;
        }
    }
}

#if defined(USE_GPU)
/**
 *  Prints all GPU SynapsesProps data.
 * 
 *  @param  allSynapsesDeviceProps   Reference to the AllSTDPSynapsesProps class on device memory.
 */
void AllSpikingSynapsesProps::printGPUSynapsesProps( void* allSynapsesDeviceProps ) 
{
    AllSpikingSynapsesProps allSynapsesProps;

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
   
    
    // copy everything
    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSpikingSynapsesProps ), cudaMemcpyDeviceToHost ) );
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
