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
 *  @param[in] allSynapsesProps     Pointer to AllSpikingSynapsesProps structures 
 *                                   on device memory.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__device__ void preSpikingSynapsesSpikeHitDevice( const BGSIZE iSyn, AllSpikingSynapsesProps* allSynapsesProps, const CLUSTER_INDEX_TYPE iCluster, int iStepOffset ) {
        allSynapsesProps->preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesProps     Pointer to AllSpikingSynapsesProps structures 
 *                                   on device memory.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__device__ void postSpikingSynapsesSpikeHitDevice( const BGSIZE iSyn, AllSpikingSynapsesProps* allSynapsesProps, int iStepOffset ) {
}

/*
 *  Prepares Synapse for a spike hit (for back propagation).
 *
 *  @param[in] iSyn                  Index of the Synapse to update.
 *  @param[in] allSynapsesProps     Pointer to AllSTDPSynapsesProps structures 
 *                                   on device memory.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__device__ void postSTDPSynapseSpikeHitDevice( const BGSIZE iSyn, AllSTDPSynapsesProps* allSynapsesProps, int iStepOffset ) {
        int total_delay = allSynapsesProps->total_delayPost[iSyn];
        allSynapsesProps->postSpikeQueue->addAnEvent(iSyn, total_delay, iStepOffset);
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
 *  @param[in] allNeuronsProps      Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesProps     Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__global__ void advanceLIFNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, SynapseIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation, int iStepOffset ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalNeurons )
                return;

        allNeuronsProps->hasFired[idx] = false;
        BGFLOAT& sp = allNeuronsProps->summation_map[idx];
        BGFLOAT& vm = allNeuronsProps->Vm[idx];
        BGFLOAT r_sp = sp;
        BGFLOAT r_vm = vm;

        if ( allNeuronsProps->nStepsInRefr[idx] > 0 ) { // is neuron refractory?
                --allNeuronsProps->nStepsInRefr[idx];
        } else if ( r_vm >= allNeuronsProps->Vthresh[idx] ) { // should it fire?
                int& spikeCount = allNeuronsProps->spikeCount[idx];
                int& spikeCountOffset = allNeuronsProps->spikeCountOffset[idx];

                // Note that the neuron has fired!
                allNeuronsProps->hasFired[idx] = true;

                // record spike time
                int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
                allNeuronsProps->spike_history[idx][idxSp] = simulationStep + iStepOffset;
                spikeCount++;

                DEBUG_SYNAPSE(
                    printf("advanceLIFNeuronsDevice\n");
                    printf("          index: %d\n", idx);
                    printf("          simulationStep: %d\n\n", simulationStep + iStepOffset);
                );

                // calculate the number of steps in the absolute refractory period
                allNeuronsProps->nStepsInRefr[idx] = static_cast<int> ( allNeuronsProps->Trefract[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsProps->Vreset[idx];

                // notify outgoing synapses of spike
                BGSIZE synapse_counts = synapseIndexMapDevice->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->outgoingSynapseBegin[idx];
                    // get the memory location of where that list begins
                    OUTGOING_SYNAPSE_INDEX_TYPE* outgoingMap_begin = &(synapseIndexMapDevice->outgoingSynapseIndexMap[beginIndex]);

                    // for each synapse, let them know we have fired
                    for (BGSIZE i = 0; i < synapse_counts; i++) {
                        OUTGOING_SYNAPSE_INDEX_TYPE idx = outgoingMap_begin[i];
                        // outgoing synapse index consists of cluster index + synapse index
                        CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
                        BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
                        preSpikingSynapsesSpikeHitDevice(iSyn, allSynapsesProps, iCluster, iStepOffset);
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
                            postSTDPSynapseSpikeHitDevice(incomingMap_begin[i], static_cast<AllSTDPSynapsesProps *>(allSynapsesProps), iStepOffset);
                        } // end for
                        break;

                    case classAllSpikingSynapses:
                    case classAllDSSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSpikingSynapsesSpikeHitDevice(incomingMap_begin[i], allSynapsesProps, iStepOffset);
                        } // end for
                        break;

                    default:
                        assert(false);
                    } // end switch
                }
        } else {
                r_sp += allNeuronsProps->I0[idx]; // add IO

                // Random number alg. goes here
                r_sp += (randNoise[idx] * allNeuronsProps->Inoise[idx]); // add cheap noise
                vm = allNeuronsProps->C1[idx] * r_vm + allNeuronsProps->C2[idx] * ( r_sp ); // decay Vm and add inputs
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
 *  @param[in] allNeuronsProps      Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesProps     Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__global__ void advanceIZHNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, SynapseIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation, int iStepOffset ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalNeurons )
                return;

        allNeuronsProps->hasFired[idx] = false;
        BGFLOAT& sp = allNeuronsProps->summation_map[idx];
        BGFLOAT& vm = allNeuronsProps->Vm[idx];
        BGFLOAT& a = allNeuronsProps->Aconst[idx];
        BGFLOAT& b = allNeuronsProps->Bconst[idx];
        BGFLOAT& u = allNeuronsProps->u[idx];
        BGFLOAT r_sp = sp;
        BGFLOAT r_vm = vm;
        BGFLOAT r_a = a;
        BGFLOAT r_b = b;
        BGFLOAT r_u = u;

        if ( allNeuronsProps->nStepsInRefr[idx] > 0 ) { // is neuron refractory?
                --allNeuronsProps->nStepsInRefr[idx];
        } else if ( r_vm >= allNeuronsProps->Vthresh[idx] ) { // should it fire?
                int& spikeCount = allNeuronsProps->spikeCount[idx];
                int& spikeCountOffset = allNeuronsProps->spikeCountOffset[idx];

                // Note that the neuron has fired!
                allNeuronsProps->hasFired[idx] = true;

                // record spike time
                int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
                allNeuronsProps->spike_history[idx][idxSp] = simulationStep + iStepOffset;
                spikeCount++;

                // calculate the number of steps in the absolute refractory period
                allNeuronsProps->nStepsInRefr[idx] = static_cast<int> ( allNeuronsProps->Trefract[idx] / deltaT + 0.5 );

                // reset to 'Vreset'
                vm = allNeuronsProps->Cconst[idx] * 0.001;
                u = r_u + allNeuronsProps->Dconst[idx];

                // notify outgoing synapses of spike
                BGSIZE synapse_counts = synapseIndexMapDevice->outgoingSynapseCount[idx];
                if (synapse_counts != 0) {
                    // get the index of where this neuron's list of synapses are
                    BGSIZE beginIndex = synapseIndexMapDevice->outgoingSynapseBegin[idx]; 
                    // get the memory location of where that list begins
                    OUTGOING_SYNAPSE_INDEX_TYPE* outgoingMap_begin = &(synapseIndexMapDevice->outgoingSynapseIndexMap[beginIndex]);
                   
                    // for each synapse, let them know we have fired
                    for (BGSIZE i = 0; i < synapse_counts; i++) {
                        OUTGOING_SYNAPSE_INDEX_TYPE idx = outgoingMap_begin[i];
                        // outgoing synapse index consists of cluster index + synapse index
                        CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
                        BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
                        preSpikingSynapsesSpikeHitDevice(iSyn, allSynapsesProps, iCluster, iStepOffset);
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
                            postSTDPSynapseSpikeHitDevice(incomingMap_begin[i], static_cast<AllSTDPSynapsesProps *>(allSynapsesProps), iStepOffset);
                        } // end for
                        break;
                    
                    case classAllSpikingSynapses:
                    case classAllDSSynapses:
                        for (BGSIZE i = 0; i < synapse_counts; i++) {
                            postSpikingSynapsesSpikeHitDevice(incomingMap_begin[i], allSynapsesProps, iStepOffset);
                        } // end for
                        break;

                    default:
                        assert(false);
                    } // end switch
                }
        } else {
                r_sp += allNeuronsProps->I0[idx]; // add IO

                // Random number alg. goes here
                r_sp += (randNoise[idx] * allNeuronsProps->Inoise[idx]); // add cheap noise

                BGFLOAT Vint = r_vm * 1000;

                // Izhikevich model integration step
                BGFLOAT Vb = Vint + allNeuronsProps->C3[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
                u = r_u + allNeuronsProps->C3[idx] * r_a * (r_b * Vint - r_u);

                vm = Vb * 0.001 + allNeuronsProps->C2[idx] * r_sp;  // add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
}
