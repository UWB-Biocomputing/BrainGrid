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
void AllLIFNeurons::advanceNeurons( IAllSynapses &synapses, IAllNeurons** allNeuronsDevice, IAllSynapses** allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap** synapseIndexMapDevice )
{
   int neuron_count;
   int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

   // CUDA parameters
   const int threadsPerBlock = 256;
   int blocksPerGrid;
    
   for(int i = 0; i < sim_info->numGPU; i++){
      cudaSetDevice(i);
      neuron_count = sim_info->individualGPUInfo[i].totalNeurons;
      blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

      // Advance neurons ------------->
      advanceLIFNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIFNeurons *)allNeuronsDevice[i], (AllSpikingSynapses*)allSynapsesDevice[i], synapseIndexMapDevice[i], (void (*)(const uint32_t, AllSpikingSynapses*))m_fpPreSpikeHit_h, (void (*)(const uint32_t, AllSpikingSynapses*))m_fpPostSpikeHit_h, m_fAllowBackPropagation);
   }
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
__global__ void advanceLIFNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, void (*fpPreSpikeHit)(const uint32_t, AllSpikingSynapses*), void (*fpPostSpikeHit)(const uint32_t, AllSpikingSynapses*), bool fAllowBackPropagation ) {
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

                //notify incomming synapses of spike
                size_t synapse_counts = allSynapsesDevice->synapse_counts[idx];
                uint32_t synapse_notified = 0;
                if(fAllowBackPropagation && synapse_counts != 0){
                   for(uint32_t synapse_index = maxSynapses * idx ; synapse_notified < synapse_counts; synapse_index++){
                      if (allSynapsesDevice->in_use[synapse_index] == true) {
                         fpPostSpikeHit(synapse_index, allSynapsesDevice); 
                         synapse_notified++;
                      }
                   }
                }

                // notify outgoing synapses of spike
                synapse_counts = synapseIndexMapDevice->synapseCount[idx];
                if(synapse_counts != 0){
                   int beginIndex = synapseIndexMapDevice->outgoingSynapse_begin[idx]; //get the index of where this neuron's list of synapses are 
                   uint32_t * forwardMap_begin = &(synapseIndexMapDevice->forwardIndex[beginIndex]); //get the memory location of where that list begins
                   
                   //for each synapse, let them know we have fired
                   for(uint32_t i = 0; i < synapse_counts; i++){
                      fpPreSpikeHit(forwardMap_begin[i], allSynapsesDevice);
                   }
                   //synapse_notified += synapse_counts; //we could increment this every time we notified a synapse, but we know how many we are going to notify, and there currently isn't a way notification could fail so this seems better
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
