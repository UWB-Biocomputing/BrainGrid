#include "AllSpikingNeurons.h"
#include "AllSpikingSynapses.h"

// Default constructor
AllSpikingNeurons::AllSpikingNeurons() : AllNeurons()
{
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

// Copy constructor
AllSpikingNeurons::AllSpikingNeurons(const AllSpikingNeurons &r_neurons) : AllNeurons(r_neurons)
{
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

AllSpikingNeurons::~AllSpikingNeurons()
{
    freeResources();
}

/*
 *  Assignment operator: copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
IAllNeurons &AllSpikingNeurons::operator=(const IAllNeurons &r_neurons)
{
    copyParameters(dynamic_cast<const AllSpikingNeurons &>(r_neurons));

    return (*this);
}

/*
 *  Copy neurons parameters.
 *
 *  @param  r_neurons  Neurons class object to copy from.
 */
void AllSpikingNeurons::copyParameters(const AllSpikingNeurons &r_neurons)
{
    AllNeurons::copyParameters(r_neurons);
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 */
void AllSpikingNeurons::setupNeurons(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
    AllNeurons::setupNeurons(sim_info, clr_info);

    // TODO: Rename variables for easier identification
    hasFired = new bool[size];
    spikeCount = new int[size];
    spikeCountOffset = new int[size];
    spike_history = new uint64_t*[size];

    for (int i = 0; i < size; ++i) {
        spike_history[i] = NULL;
        hasFired[i] = false;
        spikeCount[i] = 0;
        spikeCountOffset[i] = 0;
    }

    clr_info->pClusterSummationMap = summation_map;
}

/*
 *  Cleanup the class (deallocate memories).
 */
void AllSpikingNeurons::cleanupNeurons()
{
    freeResources();
    AllNeurons::cleanupNeurons();
}

/*
 *  Deallocate all resources
 */
void AllSpikingNeurons::freeResources()
{
    if (size != 0) {
        for(int i = 0; i < size; i++) {
            delete[] spike_history[i];
        }
    
        delete[] hasFired;
        delete[] spikeCount;
        delete[] spikeCountOffset;
        delete[] spike_history;
    }
        
    hasFired = NULL;
    spikeCount = NULL;
    spikeCountOffset = NULL;
    spike_history = NULL;
}

#if !defined(USE_GPU)
/*
 *  Clear the spike counts out of all Neurons.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 *  @param  clr_info  ClusterInfo class to read information from.
 *  @param  clr       Cluster class to read information from.
 */
void AllSpikingNeurons::clearSpikeCounts(const SimulationInfo *sim_info, const ClusterInfo *clr_info, Cluster *clr)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    for (int i = 0; i < clr_info->totalClusterNeurons; i++) {
        spikeCountOffset[i] = (spikeCount[i] + spikeCountOffset[i]) % max_spikes;
        spikeCount[i] = 0;
    }
}

/*
 *  Update internal state of the indexed Neuron (called by every simulation step).
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses         The Synapse list to search from.
 *  @param  sim_info         SimulationInfo class to read information from.
 *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
 *  @param  clr_info         ClusterInfo class to read information from.
 */
void AllSpikingNeurons::advanceNeurons(IAllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap, const ClusterInfo *clr_info)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    // For each neuron in the network
    for (int idx = clr_info->totalClusterNeurons - 1; idx >= 0; --idx) {
        // advance neurons
        advanceNeuron(idx, sim_info, clr_info);

        // notify outgoing/incomming synapses if neuron has fired
        if (hasFired[idx]) {
            DEBUG_MID(cout << " !! Neuron" << idx << "has Fired @ t: " << g_simulationStep * sim_info->deltaT << endl;)

            assert( spikeCount[idx] < max_spikes );

            if (spSynapses.total_synapse_counts != 0) {
                // notify outgoing synapses
                BGSIZE synapse_counts;

                synapse_counts = synapseIndexMap->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    BGSIZE beginIndex = synapseIndexMap->outgoingSynapseBegin[idx];
                    OUTGOING_SYNAPSE_INDEX_TYPE* outgoingMap_begin = &( synapseIndexMap->outgoingSynapseIndexMap[beginIndex] );
                    for ( BGSIZE i = 0; i < synapse_counts; i++ ) {
                        OUTGOING_SYNAPSE_INDEX_TYPE idx = outgoingMap_begin[i];
                        // outgoing synapse index consists of cluster index + synapse index
                        CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
                        BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
                        spSynapses.preSpikeHit(iSyn, iCluster);
                    }
                }

                // notify incomming synapses
                synapse_counts = synapseIndexMap->incomingSynapseIndexMap[idx];

                if (spSynapses.allowBackPropagation() && synapse_counts != 0) {
                    int beginIndex = synapseIndexMap->incomingSynapseBegin[idx];
                    BGSIZE* incomingMap_begin = &( synapseIndexMap->incomingSynapseIndexMap[beginIndex] );
          
                    for ( BGSIZE i = 0; i < synapse_counts; i++ ) {
                        BGSIZE iSyn = incomingMap_begin[i];
                        spSynapses.postSpikeHit(iSyn);
                    }
                }
            }

            hasFired[idx] = false;
        }
    }
}

/*
 *  Fire the selected Neuron and calculate the result.
 *
 *  @param  index       Index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllSpikingNeurons::fire(const int index, const SimulationInfo *sim_info) const
{
    // Note that the neuron has fired!
    hasFired[index] = true;
    
    // record spike time
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    int idxSp = (spikeCount[index] + spikeCountOffset[index]) % max_spikes;
    spike_history[index][idxSp] = g_simulationStep;

    DEBUG_SYNAPSE(
        cout << "AllSpikingNeurons::fire:" << endl;
        cout << "          index: " << index << endl;
        cout << "          g_simulationStep: " << g_simulationStep << endl << endl;
    );
    
    // increment spike count and total spike count
    spikeCount[index]++;
}

/*
 *  Get the spike history of neuron[index] at the location offIndex.
 *
 *  @param  index            Index of the neuron to get spike history.
 *  @param  offIndex         Offset of the history buffer to get from.
 *  @param  sim_info         SimulationInfo class to read information from.
 */
uint64_t AllSpikingNeurons::getSpikeHistory(int index, int offIndex, const SimulationInfo *sim_info)
{
    // offIndex is a minus offset
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    int idxSp = (spikeCount[index] + spikeCountOffset[index] +  max_spikes + offIndex) % max_spikes;
    return spike_history[index][idxSp];
}
#endif
