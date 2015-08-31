#include "AllLIFNeurons.h"
#include "Book.h"

/**
 *  Notify outgoing synapses if neuron has fired.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllLIFNeurons::advanceNeurons( AllSynapses &synapses, AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice )
{
    int neuron_count = sim_info->totalNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    bool fAllowBackPropagation = spSynapses.allowBackPropagation();
    unsigned long long fpPreSpikeHit_h = NULL;
    unsigned long long fpPostSpikeHit_h = NULL;
    spSynapses.getFpPreSpikeHit(fpPreSpikeHit_h);
    if (fAllowBackPropagation) {
        spSynapses.getFpPostSpikeHit(fpPostSpikeHit_h);
    }

    advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIFNeurons *)allNeuronsDevice, (AllSpikingSynapses*)allSynapsesDevice, synapseIndexMapDevice, (void (*)(const uint32_t, AllSpikingSynapses*))fpPreSpikeHit_h, (void (*)(const uint32_t, AllSpikingSynapses*))fpPostSpikeHit_h, fAllowBackPropagation );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

// CUDA code for advancing neurons
/**
* @param[in] totalNeurons       Number of neurons.
* @param[in] maxSynapses        Maximum number of synapses per neuron.
* @param[in] maxSpikes
* @param[in] deltaT             Inner simulation step duration.
* @param[in] simulationStep     The current simulation step.
* @param[in] randNoise          Pointer to device random noise array.
* @param[in] allNeuronsDevice   Pointer to Neuron structures in device memory.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
* @param[in] synapseIndexMap    Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
*/
__global__ void advanceNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, void (*fpPreSpikeHit)(const uint32_t, AllSpikingSynapses*), void (*fpPostSpikeHit)(const uint32_t, AllSpikingSynapses*), bool fAllowBackPropagation ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalNeurons )
                return;

        allNeuronsDevice->hasFired[idx] = false;
        BGFLOAT& sp = allNeuronsDevice->summation_map[idx];
        BGFLOAT& vm = allNeuronsDevice->Vm[idx];
        BGFLOAT r_sp = sp;
        BGFLOAT r_vm = vm;

        if ( allNeuronsDevice->nStepsInRefr[idx] > 0 ) { // is neuron refractory?
                --allNeuronsDevice->nStepsInRefr[idx];
        } else if ( r_vm >= allNeuronsDevice->Vthresh[idx] ) { // should it fire?
                int& spikeCount = allNeuronsDevice->spikeCount[idx];
                int& spikeCountOffset = allNeuronsDevice->spikeCountOffset[idx];

                // Note that the neuron has fired!
                allNeuronsDevice->hasFired[idx] = true;

                // record spike time
                int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
                allNeuronsDevice->spike_history[idx][idxSp] = simulationStep;
                spikeCount++;

                DEBUG_SYNAPSE(
                    printf("advanceNeuronsDevice\n");
                    printf("          index: %d\n", idx);
                    printf("          simulationStep: %d\n\n", simulationStep);
                );

                // calculate the number of steps in the absolute refractory period
                allNeuronsDevice->nStepsInRefr[idx] = static_cast<int> ( allNeuronsDevice->Trefract[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsDevice->Vreset[idx];

                // notify outgoing synapses of spike
                size_t synapse_counts = allSynapsesDevice->synapse_counts[idx];
                int synapse_notified = 0;
                for (int i = 0; synapse_notified < synapse_counts; i++) {
                        uint32_t iSyn = maxSynapses * idx + i;
                        if (allSynapsesDevice->in_use[iSyn] == true) {
                                fpPreSpikeHit(iSyn, allSynapsesDevice); 
                                synapse_notified++;
                        }
                }

                // notify incomming synapses of spike
                synapse_counts = synapseIndexMapDevice->synapseCount[idx];
                if (fAllowBackPropagation && synapse_counts != 0) {
                        int beginIndex = synapseIndexMapDevice->incomingSynapse_begin[idx];
                        uint32_t* inverseMap_begin = &( synapseIndexMapDevice->inverseIndex[beginIndex] );
                        uint32_t iSyn = inverseMap_begin[0];
                        for ( uint32_t i = 0; i < synapse_counts; i++ ) {
                                iSyn = inverseMap_begin[i];
                                fpPostSpikeHit(iSyn, allSynapsesDevice);
                                synapse_notified++;
                        }
                }
        } else {
                r_sp += allNeuronsDevice->I0[idx]; // add IO

                // Random number alg. goes here
                r_sp += (randNoise[idx] * allNeuronsDevice->Inoise[idx]); // add cheap noise
                vm = allNeuronsDevice->C1[idx] * r_vm + allNeuronsDevice->C2[idx] * ( r_sp ); // decay Vm and add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
}
