#include "AllSTDPSynapsesProps.h"
#include "EventQueue.h"
#if defined(USE_GPU)
#include <helper_cuda.h>
#endif

// Default constructor
AllSTDPSynapsesProps::AllSTDPSynapsesProps()
{
    total_delayPost = NULL;
    tauspost = NULL;
    tauspre = NULL;
    taupos = NULL;
    tauneg = NULL;
    STDPgap = NULL;
    Wex = NULL;
    Aneg = NULL;
    Apos = NULL;
    mupos = NULL;
    muneg = NULL;
    useFroemkeDanSTDP = NULL;
    postSpikeQueue = NULL;
}

AllSTDPSynapsesProps::~AllSTDPSynapsesProps()
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
void AllSTDPSynapsesProps::setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllSpikingSynapsesProps::setupSynapsesProps(num_neurons, max_synapses, sim_info, clr_info);

    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        total_delayPost = new int[max_total_synapses];
        tauspost = new BGFLOAT[max_total_synapses];
        tauspre = new BGFLOAT[max_total_synapses];
        taupos = new BGFLOAT[max_total_synapses];
        tauneg = new BGFLOAT[max_total_synapses];
        STDPgap = new BGFLOAT[max_total_synapses];
        Wex = new BGFLOAT[max_total_synapses];
        Aneg = new BGFLOAT[max_total_synapses];
        Apos = new BGFLOAT[max_total_synapses];
        mupos = new BGFLOAT[max_total_synapses];
        muneg = new BGFLOAT[max_total_synapses];
        useFroemkeDanSTDP = new bool[max_total_synapses];

        // create a post synapse spike queue & initialize it
        postSpikeQueue = new EventQueue();
#if defined(USE_GPU)
        // initializes the post synapse spike queue
        postSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses, (int)0, (int)0);
#else // USE_GPU
        // initializes the post synapse spike queue
        postSpikeQueue->initEventQueue(clr_info->clusterID, max_total_synapses);
#endif // USE_GPU
    }
}

/*
 *  Cleanup the class.
 *  Deallocate memories.
 */
void AllSTDPSynapsesProps::cleanupSynapsesProps()
{
    BGSIZE max_total_synapses = maxSynapsesPerNeuron * count_neurons;

    if (max_total_synapses != 0) {
        delete[] total_delayPost;
        delete[] tauspost;
        delete[] tauspre;
        delete[] taupos;
        delete[] tauneg;
        delete[] STDPgap;
        delete[] Wex;
        delete[] Aneg;
        delete[] Apos;
        delete[] mupos;
        delete[] muneg;
        delete[] useFroemkeDanSTDP;
    }

    total_delayPost = NULL;
    tauspost = NULL;
    tauspre = NULL;
    taupos = NULL;
    tauneg = NULL;
    STDPgap = NULL;
    Wex = NULL;
    Aneg = NULL;
    Apos = NULL;
    mupos = NULL;
    muneg = NULL;
    useFroemkeDanSTDP = NULL;

    if (postSpikeQueue != NULL) {
        delete postSpikeQueue;
        postSpikeQueue = NULL;
    }
}

#if defined(USE_GPU)
/*
 *  Allocate GPU memories to store all synapses' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSTDPSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSTDPSynapsesProps::setupSynapsesDeviceProps( void** allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron )
{
    AllSTDPSynapsesProps allSynapsesProps;

    allocSynapsesDeviceProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    checkCudaErrors( cudaMalloc( allSynapsesDeviceProps, sizeof( AllSTDPSynapsesProps ) ) );
    checkCudaErrors( cudaMemcpy ( *allSynapsesDeviceProps, &allSynapsesProps, sizeof( AllSTDPSynapsesProps ), cudaMemcpyHostToDevice ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;
}

/*
 *  Allocate GPU memories to store all synapses' states.
 *
 *  @param  allSynapsesProps      Reference to the AllSTDPSynapsesProps class.
 *  @param  num_neurons           Number of neurons.
 *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
 */
void AllSTDPSynapsesProps::allocSynapsesDeviceProps( AllSTDPSynapsesProps &allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSpikingSynapsesProps::allocSynapsesDeviceProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron);

    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.total_delayPost, size * sizeof( int ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.tauspost, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.tauspre, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.taupos, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.tauneg, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.STDPgap, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.Wex, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.Aneg, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.Apos, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.mupos, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.muneg, size * sizeof( BGFLOAT ) ) );
    checkCudaErrors( cudaMalloc( ( void ** ) &allSynapsesProps.useFroemkeDanSTDP, size * sizeof( bool ) ) );

    // create a EventQueue objet in device memory and set the pointer in device
    postSpikeQueue->createEventQueueInDevice(&allSynapsesProps.postSpikeQueue);
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesDeviceProps  Reference to the AllSTDPSynapsesProps class on device memory.
 */
void AllSTDPSynapsesProps::cleanupSynapsesDeviceProps( void* allSynapsesDeviceProps )
{
    AllSTDPSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    deleteSynapsesDeviceProps( allSynapsesProps );

    checkCudaErrors( cudaFree( allSynapsesDeviceProps ) );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSTDPSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allSynapsesProps  Reference to the AllSTDPSynapsesProps class.
 */
void AllSTDPSynapsesProps::deleteSynapsesDeviceProps( AllSTDPSynapsesProps& allSynapsesProps )
{
    checkCudaErrors( cudaFree( allSynapsesProps.total_delayPost ) );
    checkCudaErrors( cudaFree( allSynapsesProps.tauspost ) );
    checkCudaErrors( cudaFree( allSynapsesProps.tauspre ) );
    checkCudaErrors( cudaFree( allSynapsesProps.taupos ) );
    checkCudaErrors( cudaFree( allSynapsesProps.tauneg ) );
    checkCudaErrors( cudaFree( allSynapsesProps.STDPgap ) );
    checkCudaErrors( cudaFree( allSynapsesProps.Wex ) );
    checkCudaErrors( cudaFree( allSynapsesProps.Aneg ) );
    checkCudaErrors( cudaFree( allSynapsesProps.Apos ) );
    checkCudaErrors( cudaFree( allSynapsesProps.mupos ) );
    checkCudaErrors( cudaFree( allSynapsesProps.muneg ) );
    checkCudaErrors( cudaFree( allSynapsesProps.useFroemkeDanSTDP ) );

    // delete EventQueue object in device memory.
    EventQueue::deleteEventQueueInDevice(allSynapsesProps.postSpikeQueue);

    AllSpikingSynapsesProps::deleteSynapsesDeviceProps( allSynapsesProps );
}

/*
 *  Copy all synapses' data from host to device.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSTDPSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSTDPSynapsesProps::copySynapseHostToDeviceProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron )
{
    AllSTDPSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    copyHostToDeviceProps( allSynapsesDeviceProps, allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSTDPSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Copy all synapses' data from host to device.
 *  (Helper function of copySynapseHostToDeviceProps)
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSTDPSynapsesProps class on device memory.
 *  @param  allSynapsesProps         Reference to the AllSTDPSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSTDPSynapsesProps::copyHostToDeviceProps( void* allSynapsesDeviceProps, AllSTDPSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron )
{
    // copy everything necessary
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSpikingSynapsesProps::copyHostToDeviceProps( allSynapsesDeviceProps, allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    checkCudaErrors( cudaMemcpy ( allSynapsesProps.total_delayPost, total_delayPost,
            size * sizeof( int ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.tauspost, tauspost,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.tauspre, tauspre,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.taupos, taupos,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.tauneg, tauneg,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.STDPgap, STDPgap,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.Wex, Wex,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.Aneg, Aneg,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.Apos, Apos,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.mupos, mupos,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.muneg, muneg,
            size * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy ( allSynapsesProps.useFroemkeDanSTDP, useFroemkeDanSTDP,
            size * sizeof( bool ), cudaMemcpyHostToDevice ) );

    // copy event queue data from host to device.
    postSpikeQueue->copyEventQueueHostToDevice(allSynapsesProps.postSpikeQueue);
}

/*
 *  Copy all synapses' data from device to host.
 *
 *  @param  allSynapsesDeviceProps   Reference to the AllSTDPSynapsesProps class on device memory.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSTDPSynapsesProps::copySynapseDeviceToHostProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron )
{
    AllSTDPSynapsesProps allSynapsesProps;

    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    copyDeviceToHostProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron );

    // The preSpikeQueue points to an EventQueue objet in device memory. The pointer is copied to allSynapsesDeviceProps.
    // To avoide illegeal deletion of the object at AllSpikingSynapsesProps::cleanupSynapsesProps(), set the pointer to NULL.
    allSynapsesProps.preSpikeQueue = NULL;

    // Set count_neurons to 0 to avoid illegal memory deallocation
    // at AllSTDPSynapsesProps deconstructor.
    allSynapsesProps.count_neurons = 0;
}

/*
 *  Copy all synapses' data from device to host.
 *  (Helper function of copySynapseDeviceToHostProps)
 *
 *  @param  allSynapsesProps         Reference to the AllSTDPSynapsesProps class.
 *  @param  num_neurons              Number of neurons.
 *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
 */
void AllSTDPSynapsesProps::copyDeviceToHostProps( AllSTDPSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron)
{
    BGSIZE size = maxSynapsesPerNeuron * num_neurons;

    AllSpikingSynapsesProps::copyDeviceToHostProps( allSynapsesProps, num_neurons, maxSynapsesPerNeuron);

    checkCudaErrors( cudaMemcpy ( tauspost, allSynapsesProps.tauspost,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( tauspre, allSynapsesProps.tauspre,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( taupos, allSynapsesProps.taupos,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( tauneg, allSynapsesProps.tauneg,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( STDPgap, allSynapsesProps.STDPgap,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Wex, allSynapsesProps.Wex,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Aneg, allSynapsesProps.Aneg,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( Apos, allSynapsesProps.Apos,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( mupos, allSynapsesProps.mupos,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( muneg, allSynapsesProps.muneg,
            size * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
    checkCudaErrors( cudaMemcpy ( useFroemkeDanSTDP, allSynapsesProps.useFroemkeDanSTDP,
            size * sizeof( bool ), cudaMemcpyDeviceToHost ) );

    // copy event queue data from device to host.
    postSpikeQueue->copyEventQueueDeviceToHost(allSynapsesProps.postSpikeQueue);
}
#endif // USE_GPU

/*
 *  Sets the data for Synapse to input's data.
 *
 *  @param  input  istream to read from.
 *  @param  iSyn   Index of the synapse to set.
 */
void AllSTDPSynapsesProps::readSynapseProps(istream &input, const BGSIZE iSyn)
{
    AllSpikingSynapsesProps::readSynapseProps(input, iSyn);

    // input.ignore() so input skips over end-of-line characters.
    input >> total_delayPost[iSyn]; input.ignore();
    input >> tauspost[iSyn]; input.ignore();
    input >> tauspre[iSyn]; input.ignore();
    input >> taupos[iSyn]; input.ignore();
    input >> tauneg[iSyn]; input.ignore();
    input >> STDPgap[iSyn]; input.ignore();
    input >> Wex[iSyn]; input.ignore();
    input >> Aneg[iSyn]; input.ignore();
    input >> Apos[iSyn]; input.ignore();
    input >> mupos[iSyn]; input.ignore();
    input >> muneg[iSyn]; input.ignore();
    input >> useFroemkeDanSTDP[iSyn]; input.ignore();
}

/*
 *  Write the synapse data to the stream.
 *
 *  @param  output  stream to print out to.
 *  @param  iSyn    Index of the synapse to print out.
 */
void AllSTDPSynapsesProps::writeSynapseProps(ostream& output, const BGSIZE iSyn) const
{
    AllSpikingSynapsesProps::writeSynapseProps(output, iSyn);

    output << total_delayPost[iSyn] << ends;
    output << tauspost[iSyn] << ends;
    output << tauspre[iSyn] << ends;
    output << taupos[iSyn] << ends;
    output << tauneg[iSyn] << ends;
    output << STDPgap[iSyn] << ends;
    output << Wex[iSyn] << ends;
    output << Aneg[iSyn] << ends;
    output << Apos[iSyn] << ends;
    output << mupos[iSyn] << ends;
    output << muneg[iSyn] << ends;
    output << useFroemkeDanSTDP[iSyn] << ends;
}

/*
 *  Prints all SynapsesProps data.
 */
void AllSTDPSynapsesProps::printSynapsesProps() 
{
    AllSpikingSynapsesProps::printSynapsesProps();
    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        if (W[i] != 0.0) {
            cout << "total_delayPost: " << total_delayPost[i];
            cout << " tauspost: " << tauspost[i];
            cout << " tauspre: " << tauspre[i];
            cout << " taupos: " << taupos[i];
            cout << " tauneg: " << tauneg[i];
            cout << " STDPgap: " << STDPgap[i];
            cout << " Wex: " << Wex[i];
            cout << " Aneg: " << Aneg[i];
            cout << " Apos: " << Apos[i];
            cout << " mupos: " << mupos[i];
            cout << " muneg: " << muneg[i];
            cout << " useFroemkeDanSTDP: " << useFroemkeDanSTDP[i];
            cout << " postSpikeQueue: " << postSpikeQueue->m_queueEvent[i] << endl;
        } else {
            if(total_delayPost[i] != 0.0 || tauspost[i] != 0.0 || 
            tauspre[i] != 0.0 || taupos[i] != 0.0 ||
            tauneg[i] != 0.0 || STDPgap[i] != 0.0 ||
            Wex[i] != 0.0 || Aneg[i] != 0.0 ||
            Apos[i] != 0.0 || mupos[i] ||
            muneg[i] != 0.0 || useFroemkeDanSTDP[i] != 0.0 ||
            postSpikeQueue->m_queueEvent[i] != 0.0) {
                cout << "---------------------ERROR!!!!!!!!-------------";
                cout << " total_delayPost: " << total_delayPost[i];
                cout << " tauspost: " << tauspost[i];
                cout << " tauspre: " << tauspre[i];
                cout << " taupos: " << taupos[i];
                cout << " tauneg: " << tauneg[i];
                cout << " STDPgap: " << STDPgap[i];
                cout << " Wex: " << Wex[i];
                cout << " Aneg: " << Aneg[i];
                cout << " Apos: " << Apos[i];
                cout << " mupos: " << mupos[i];
                cout << " muneg: " << muneg[i];
                cout << " useFroemkeDanSTDP: " << useFroemkeDanSTDP[i];
                cout << " postSpikeQueue: " << postSpikeQueue->m_queueEvent[i] << endl;
            }
        }
    }

}

#if defined(USE_GPU)
void AllSTDPSynapsesProps::printGPUSynapsesProps( void* allSynapsesDeviceProps ) 
{
    AllSTDPSynapsesProps allSynapsesProps;
    checkCudaErrors( cudaMemcpy ( &allSynapsesProps, allSynapsesDeviceProps, sizeof( AllSTDPSynapsesProps ), cudaMemcpyDeviceToHost ) );
    printGPUSynapsesPropsHelper( allSynapsesProps );
}

void AllSTDPSynapsesProps::printGPUSynapsesPropsHelper( AllSynapsesProps& allSynapsesProps )
{
    cout << "This is GPU SynapsesProps data:" << endl;
    AllSpikingSynapsesProps::printGPUSynapsesPropsHelper( allSynapsesProps );
}
#endif // USE_GPU
