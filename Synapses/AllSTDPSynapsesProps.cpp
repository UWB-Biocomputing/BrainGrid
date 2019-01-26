#include "AllSTDPSynapsesProps.h"
#include "EventQueue.h"

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
