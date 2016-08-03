#include "AllLIFNeurons.h"
#include "Book.h"

/*
 *  Update the state of all neurons for a time step
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  synapses               Reference to the allSynapses struct on host memory.
 *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
 *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
 *  @param  sim_info               SimulationInfo to refer from.
 *  @param  randNoise              Reference to the random noise array.
 *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
 */
void AllLIFNeurons::advanceNeurons( IAllSynapses &synapses, IAllNeurons* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice )
{
    int neuron_count = sim_info->totalNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceLIFNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIFNeurons *)allNeuronsDevice, (AllSpikingSynapsesDeviceProperties*)allSynapsesDevice, synapseIndexMapDevice, (void (*)(const BGSIZE, AllSpikingSynapsesDeviceProperties*))m_fpPreSpikeHit_h, (void (*)(const BGSIZE, AllSpikingSynapsesDeviceProperties*))m_fpPostSpikeHit_h, m_fAllowBackPropagation );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

/*
 *  CUDA code for advancing LIF neurons
 * 
 *  @param[in] totalNeurons          Number of neurons.
 *  @param[in] maxSynapses           Maximum number of synapses per neuron.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] randNoise             Pointer to device random noise array.
 *  @param[in] allNeuronsDevice      Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 *  @param[in] fpPreSpikeHit         Pointer to the device function preSpikeHit() function.
 *  @param[in] fpPostSpikeHit        Pointer to the device function postSpikeHit() function.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 */
__global__ void advanceLIFNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeurons* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, void (*fpPreSpikeHit)(const BGSIZE, AllSpikingSynapsesDeviceProperties*), void (*fpPostSpikeHit)(const BGSIZE, AllSpikingSynapsesDeviceProperties*), bool fAllowBackPropagation ) {
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
                    printf("advanceLIFNeuronsDevice\n");
                    printf("          index: %d\n", idx);
                    printf("          simulationStep: %d\n\n", simulationStep);
                );

                // calculate the number of steps in the absolute refractory period
                allNeuronsDevice->nStepsInRefr[idx] = static_cast<int> ( allNeuronsDevice->Trefract[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsDevice->Vreset[idx];

                // notify outgoing synapses of spike
                BGSIZE synapse_counts = allSynapsesDevice->synapse_counts[idx];
                int synapse_notified = 0;
                for (BGSIZE i = 0; synapse_notified < synapse_counts; i++) {
                        BGSIZE iSyn = maxSynapses * idx + i;
                        if (allSynapsesDevice->in_use[iSyn] == true) {
                                fpPreSpikeHit(iSyn, allSynapsesDevice); 
                                synapse_notified++;
                        }
                }

                // notify incomming synapses of spike
                synapse_counts = synapseIndexMapDevice->synapseCount[idx];
                if (fAllowBackPropagation && synapse_counts != 0) {
                        BGSIZE beginIndex = synapseIndexMapDevice->incomingSynapse_begin[idx];
                        BGSIZE* inverseMap_begin = &( synapseIndexMapDevice->inverseIndex[beginIndex] );
                        BGSIZE iSyn = inverseMap_begin[0];
                        for ( BGSIZE i = 0; i < synapse_counts; i++ ) {
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
