/*
 * AllNeuronsDeviceFuncs_d.cu
 *
 */

#include "AllNeuronsDeviceFuncs.h"
#include "AllSynapsesDeviceFuncs.h"

/* -------------------------------------*\
|* # Device Functions for advanceNeurons
\* -------------------------------------*/

/*
 *  Prepares Synapse for a spike hit.
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
 *                                   on device memory.
 */
__device__ void preSpikingSynapsesSpikeHitDevice( const BGSIZE iSyn, AllSpikingSynapsesDeviceProperties* allSynapsesDevice ) {
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
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesDevice     Pointer to AllSpikingSynapsesDeviceProperties structures 
 *                                   on device memory.
 */
__device__ void postSpikingSynapsesSpikeHitDevice( const BGSIZE iSyn, AllSpikingSynapsesDeviceProperties* allSynapsesDevice ) {
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesDevice     Pointer to AllSTDPSynapsesDeviceProperties structures 
 *                                   on device memory.
 */
__device__ void postSTDPSynapseSpikeHitDevice( const BGSIZE iSyn, AllSTDPSynapsesDeviceProperties* allSynapsesDevice ) {
        uint32_t &delay_queue = allSynapsesDevice->delayQueuePost[iSyn];
        int delayIdx = allSynapsesDevice->delayIdxPost[iSyn];
        int ldelayQueue = allSynapsesDevice->ldelayQueuePost[iSyn];
        int total_delay = allSynapsesDevice->total_delayPost[iSyn];

        // Add to spike queue

        // calculate index where to insert the spike into delayQueue
        int idx = delayIdx +  total_delay;
        if ( idx >= ldelayQueue ) {
                idx -= ldelayQueue;
        }

        // set a spike
        //assert( !(delay_queue[0] & (0x1 << idx)) );
        delay_queue |= (0x1 << idx);
}

/* -------------------------------------*\
|* # Global Functions for advanceNeurons
\* -------------------------------------*/

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
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 */
__global__ void advanceLIFNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation ) {
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
                BGSIZE synapse_counts = synapseIndexMapDevice->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->outgoingSynapseBegin[idx];
                    // get the memory location of where that list begins
                    BGSIZE* outgoingMap_begin = &(synapseIndexMapDevice->outgoingSynapseIndexMap[beginIndex]);

                    // for each synapse, let them know we have fired
                    for (BGSIZE i = 0; i < synapse_counts; i++) {
                        preSpikingSynapsesSpikeHitDevice(outgoingMap_begin[i], allSynapsesDevice);
                    }
                }

                // notify incomming synapses of spike
                synapse_counts = synapseIndexMapDevice->incomingSynapseCount[idx];
                if (fAllowBackPropagation && synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->incomingSynapseBegin[idx];
                    // get the memory location of where that list begins
                    BGSIZE* incomingMap_begin = &(synapseIndexMapDevice->incomingSynapseIndexMap[beginIndex]);

                    // for each synapse, let them know we have fired
                    switch (classSynapses_d) {
                    case classAllSTDPSynapses:
                    case classAllDynamicSTDPSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSTDPSynapseSpikeHitDevice(incomingMap_begin[i], static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice));
                        } // end for
                        break;

                    case classAllSpikingSynapses:
                    case classAllDSSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSpikingSynapsesSpikeHitDevice(incomingMap_begin[i], allSynapsesDevice);
                        } // end for
                        break;

                    default:
                        assert(false);
                    } // end switch
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

/*
 *  CUDA code for advancing izhikevich neurons
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
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 */
__global__ void advanceIZHNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeuronsDeviceProperties* allNeuronsDevice, AllSpikingSynapsesDeviceProperties* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalNeurons )
                return;

        allNeuronsDevice->hasFired[idx] = false;
        BGFLOAT& sp = allNeuronsDevice->summation_map[idx];
        BGFLOAT& vm = allNeuronsDevice->Vm[idx];
        BGFLOAT& a = allNeuronsDevice->Aconst[idx];
        BGFLOAT& b = allNeuronsDevice->Bconst[idx];
        BGFLOAT& u = allNeuronsDevice->u[idx];
        BGFLOAT r_sp = sp;
        BGFLOAT r_vm = vm;
        BGFLOAT r_a = a;
        BGFLOAT r_b = b;
        BGFLOAT r_u = u;

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

                // calculate the number of steps in the absolute refractory period
                allNeuronsDevice->nStepsInRefr[idx] = static_cast<int> ( allNeuronsDevice->Trefract[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsDevice->Cconst[idx] * 0.001;
                u = r_u + allNeuronsDevice->Dconst[idx];

                // notify outgoing synapses of spike
                BGSIZE synapse_counts = synapseIndexMapDevice->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->outgoingSynapseBegin[idx]; 
                    // get the memory location of where that list begins
                    BGSIZE* outgoingMap_begin = &(synapseIndexMapDevice->outgoingSynapseIndexMap[beginIndex]);
                   
                    // for each synapse, let them know we have fired
                    for (BGSIZE i = 0; i < synapse_counts; i++) {
                        preSpikingSynapsesSpikeHitDevice(outgoingMap_begin[i], allSynapsesDevice);
                    }
                }

                // notify incomming synapses of spike
                synapse_counts = synapseIndexMapDevice->incomingSynapseCount[idx];
                if (fAllowBackPropagation && synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->incomingSynapseBegin[idx];
                    // get the memory location of where that list begins
                    BGSIZE* incomingMap_begin = &(synapseIndexMapDevice->incomingSynapseIndexMap[beginIndex]);

                    // for each synapse, let them know we have fired
                    switch (classSynapses_d) {
                    case classAllSTDPSynapses:
                    case classAllDynamicSTDPSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSTDPSynapseSpikeHitDevice(incomingMap_begin[i], static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice));
                        } // end for
                        break;
                    
                    case classAllSpikingSynapses:
                    case classAllDSSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSpikingSynapsesSpikeHitDevice(incomingMap_begin[i], allSynapsesDevice);
                        } // end for
                        break;

                    default:
                        assert(false);
                    } // end switch
                }
        } else {
                r_sp += allNeuronsDevice->I0[idx]; // add IO

                // Random number alg. goes here
                r_sp += (randNoise[idx] * allNeuronsDevice->Inoise[idx]); // add cheap noise

                BGFLOAT Vint = r_vm * 1000;

                // Izhikevich model integration step
                BGFLOAT Vb = Vint + allNeuronsDevice->C3[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
                u = r_u + allNeuronsDevice->C3[idx] * r_a * (r_b * Vint - r_u);

                vm = Vb * 0.001 + allNeuronsDevice->C2[idx] * r_sp;  // add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
}
