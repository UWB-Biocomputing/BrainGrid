#include "AllLIFNeurons.h"
#include "AllDSSynapses.h"
#include "Book.h"

//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice );

/**
 *  Notify outgoing synapses if neuron has fired.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllLIFNeurons::advanceNeurons( AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise)
{
    int neuron_count = sim_info->totalNeurons;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, g_simulationStep, sim_info->maxSynapsesPerNeuron, sim_info->deltaT, randNoise, (AllIFNeurons *)allNeuronsDevice, (AllDSSynapses*)allSynapsesDevice );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

// CUDA code for advancing neurons
/**
* @param[in] totalNeurons       Number of neurons.
* @param[in] simulationStep     The current simulation step.
* @param[in] maxSynapses        Maximum number of synapses per neuron.
* @param[in] deltaT             Inner simulation step duration.
* @param[in] randNoise          Pointer to device random noise array.
* @param[in] allNeuronsDevice   Pointer to Neuron structures in device memory.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice ) {
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
                // Note that the neuron has fired!
                allNeuronsDevice->hasFired[idx] = true;

                // record spike time
                allNeuronsDevice->spike_history[idx][allNeuronsDevice->spikeCount[idx]] = simulationStep;
                allNeuronsDevice->spikeCount[idx]++;

                // calculate the number of steps in the absolute refractory period
                allNeuronsDevice->nStepsInRefr[idx] = static_cast<int> ( allNeuronsDevice->Trefract[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsDevice->Vreset[idx];

                // notify synapses of spike
                size_t synapse_counts = allSynapsesDevice->synapse_counts[idx];
                int synapse_notified = 0;
                for (int i = 0; synapse_notified < synapse_counts; i++) {
                        uint32_t iSyn = maxSynapses * idx + i;
                        if (allSynapsesDevice->in_use[iSyn] == true) {
                                uint32_t &delay_queue = allSynapsesDevice->delayQueue[iSyn];
                                int delayIdx = allSynapsesDevice->delayIdx[iSyn];
                                int ldelayQueue = allSynapsesDevice->ldelayQueue[iSyn];
                                int total_delay = allSynapsesDevice->total_delay[iSyn];

                                // Add to spike queue

                                // calculate index where to insert the spike into delayQueue
                                int idx = delayIdx +  total_delay;
                                if ( idx >= ldelayQueue ) {
                                        idx -= ldelayQueue;
                                }

                                // set a spike
                                //assert( !(delay_queue[0] & (0x1 << idx)) );
                                delay_queue |= (0x1 << idx);

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

