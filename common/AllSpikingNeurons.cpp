#include "AllSpikingNeurons.h"
#include "AllSynapses.h"

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

void AllSpikingNeurons::cleanupNeurons()
{
    freeResources();
    AllNeurons::cleanupNeurons();
}

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

/**
 *  Clear the spike counts out of all Neurons.
 */
//! Clear spike count of each neuron.
void AllSpikingNeurons::clearSpikeCounts(const SimulationInfo *sim_info)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    for (int i = 0; i < sim_info->totalNeurons; i++) {
        spikeCountOffset[i] = (spikeCount[i] + spikeCountOffset[i]) % max_spikes;
        spikeCount[i] = 0;
    }
}

#if !defined(USE_GPU)
/**
 *  Notify outgoing synapses if neuron has fired.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllSpikingNeurons::advanceNeurons(AllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap)
{
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));

    // For each neuron in the network
    for (int idx = sim_info->totalNeurons - 1; idx >= 0; --idx) {
        // advance neurons
        advanceNeuron(idx, sim_info);

        // notify outgoing/incomming synapses if neuron has fired
        if (hasFired[idx]) {
            DEBUG_MID(cout << " !! Neuron" << idx << "has Fired @ t: " << g_simulationStep * sim_info->deltaT << endl;)

            assert( spikeCount[idx] < max_spikes );

            // notify outgoing synapses
            size_t synapse_counts = synapses.synapse_counts[idx];
            int synapse_notified = 0;
            for (int z = 0; synapse_notified < synapse_counts; z++) {
                uint32_t iSyn = sim_info->maxSynapsesPerNeuron * idx + z;
                if (synapses.in_use[iSyn] == true) {
                    synapses.preSpikeHit(iSyn);
                    synapse_notified++;
                }
            }

            // notify incomming synapses
            if (synapses.allowBackPropagation()) {
                synapse_counts = synapseIndexMap->synapseCount[idx];
                if (synapse_counts != 0) {
                        int beginIndex = synapseIndexMap->incomingSynapse_begin[idx];
                        uint32_t* inverseMap_begin = &( synapseIndexMap->inverseIndex[beginIndex] );
                        uint32_t iSyn;
                        for ( uint32_t i = 0; i < synapse_counts; i++ ) {
                            iSyn = inverseMap_begin[i];
                            synapses.postSpikeHit(iSyn);
                            synapse_notified++;
                        }
                }
            }

            hasFired[idx] = false;
        }
    }
}

/**
 *  Fire the selected Neuron and calculate the result.
 *  @param  index   index of the Neuron to update.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllSpikingNeurons::fire(const int index, const SimulationInfo *sim_info) const
{
    const BGFLOAT deltaT = sim_info->deltaT;
    // Note that the neuron has fired!
    hasFired[index] = true;
    
    // record spike time
    int max_spikes = (int) ((sim_info->epochDuration * sim_info->maxFiringRate));
    int idxSp = (spikeCount[index] + spikeCountOffset[index]) % max_spikes;
    spike_history[index][idxSp] = g_simulationStep;
    
    // increment spike count and total spike count
    spikeCount[index]++;
}
#endif
