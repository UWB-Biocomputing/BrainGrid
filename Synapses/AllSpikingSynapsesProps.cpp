#include "AllSpikingSynapsesProps.h"
#include "EventQueue.h"

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
