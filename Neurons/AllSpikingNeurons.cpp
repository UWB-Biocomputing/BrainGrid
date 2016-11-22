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

AllSpikingNeurons::~AllSpikingNeurons()
{
    freeResources();
}

/*
 *  Setup the internal structure of the class (allocate memories).
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSpikingNeurons::setupNeurons(SimulationInfo *sim_info)
{
    AllNeurons::setupNeurons(sim_info);

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

    sim_info->pSummationMap = summation_map;
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

/*
 *  Clear the spike counts out of all Neurons.
 *
 *  @param  sim_info  SimulationInfo class to read information from.
 */
void AllSpikingNeurons::clearSpikeCounts(const SimulationInfo *sim_info)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    for (int i = 0; i < sim_info->totalNeurons; i++) {
        spikeCountOffset[i] = (spikeCount[i] + spikeCountOffset[i]) % max_spikes;
        spikeCount[i] = 0;
    }
}

#if !defined(USE_GPU)
/*
 *  Update internal state of the indexed Neuron (called by every simulation step).
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses         The Synapse list to search from.
 *  @param  sim_info         SimulationInfo class to read information from.
 *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
 */
void AllSpikingNeurons::advanceNeurons(IAllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    // For each neuron in the network
    for (int idx = sim_info->totalNeurons - 1; idx >= 0; --idx) {
        // advance neurons
        advanceNeuron(idx, sim_info);

        // notify outgoing/incomming synapses if neuron has fired
        if (hasFired[idx]) {
            DEBUG_MID(cout << " !! Neuron" << idx << "has Fired @ t: " << g_simulationStep * sim_info->deltaT << endl;)

            assert( spikeCount[idx] < max_spikes );

            // notify outgoing synapses
            BGSIZE synapse_counts;

            if(synapseIndexMap != NULL){
                synapse_counts = synapseIndexMap->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    int beginIndex = synapseIndexMap->outgoingSynapseBegin[idx];
                    BGSIZE* outgoingMap_begin = &( synapseIndexMap->outgoingSynapseIndexMap[beginIndex] );
                    BGSIZE iSyn;
                    for ( BGSIZE i = 0; i < synapse_counts; i++ ) {
                        iSyn = outgoingMap_begin[i];
                        spSynapses.preSpikeHit(iSyn);
                    }
                }
            }

            // notify incomming synapses
            synapse_counts = spSynapses.synapse_counts[idx];
            BGSIZE synapse_notified = 0;

            if (spSynapses.allowBackPropagation()) {
                for (int z = 0; synapse_notified < synapse_counts; z++) {
                     BGSIZE iSyn = sim_info->maxSynapsesPerNeuron * idx + z;
                     if (spSynapses.in_use[iSyn] == true) {
                         spSynapses.postSpikeHit(iSyn);
                         synapse_notified++;
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
