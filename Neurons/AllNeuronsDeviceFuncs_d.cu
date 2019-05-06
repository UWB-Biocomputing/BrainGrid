#include "AllNeuronsDeviceFuncs.h"

/* -------------------------------------*\
|* # Global Functions for advanceNeurons
\* -------------------------------------*/

/*
 *  CUDA code for advancing LIF neurons
 * 
 *  @param[in] totalNeurons          Number of neurons.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] randNoise             Pointer to device random noise array.
 *  @param[in] allNeuronsDevice      Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses 
 *                                   that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__global__ void advanceLIFNeuronsDevice(const int totalNeurons, const int maxSpikes, const BGFLOAT deltaT, 
                                        const uint64_t simulationStep, const float* randNoise, 
                                        AllIFNeuronsDeviceProperties* allNeuronsDevice, 
                                        AllSpikingSynapsesDeviceProperties* allSynapsesDevice, 
                                        SynapseIndexMap* synapseIndexMapDevice, const bool fAllowBackPropagation, 
                                        const int iStepOffset) {
    // determine which neuron this thread is processing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNeurons) return;

    // clear synaptic input for next time step
    allNeuronsDevice->summation_map[idx] = 0;

	if (allNeuronsDevice->nStepsInRefr[idx] > 0) { // is neuron refractory?
		allNeuronsDevice->hasFired[idx] = false;
		--allNeuronsDevice->nStepsInRefr[idx];
		return;
	}

	BGFLOAT& vm = allNeuronsDevice->Vm[idx];

    if ( vm >= allNeuronsDevice->Vthresh[idx] ) { // should it fire?
        int& spikeCount = allNeuronsDevice->spikeCount[idx];
        int spikeCountOffset = allNeuronsDevice->spikeCountOffset[idx];

        // Note that the neuron has fired!
        allNeuronsDevice->hasFired[idx] = true;

        // record spike time
        int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
        allNeuronsDevice->spike_history[idx][idxSp] = simulationStep + iStepOffset;
        spikeCount++;

        DEBUG_SYNAPSE(
            printf("advanceLIFNeuronsDevice\n");
            printf("          index: %d\n", idx);
            printf("          simulationStep: %d\n\n", simulationStep + iStepOffset);
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
            OUTGOING_SYNAPSE_INDEX_TYPE* outgoingMap_begin = &(synapseIndexMapDevice->outgoingSynapseIndexMap[beginIndex]);

            // for each synapse, let them know we have fired
            for (BGSIZE i = 0; i < synapse_counts; i++) {
                OUTGOING_SYNAPSE_INDEX_TYPE idx = outgoingMap_begin[i];
                // outgoing synapse index consists of cluster index + synapse index
                CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
                BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
				allSynapsesDevice->preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
            }
        }

        switch (classSynapses_d) {
        case classAllSTDPSynapses:
        case classAllDynamicSTDPSynapses:
            // notify incomming synapses of spike
            synapse_counts = synapseIndexMapDevice->incomingSynapseCount[idx];
            if (fAllowBackPropagation && synapse_counts != 0) {
                // get the index of where this neuron's list of synapses are
                BGSIZE beginIndex = synapseIndexMapDevice->incomingSynapseBegin[idx];
                // get the memory location of where that list begins
                BGSIZE* incomingMap_begin = &(synapseIndexMapDevice->incomingSynapseIndexMap[beginIndex]);

                AllSTDPSynapsesDeviceProperties * synapseDevice = static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice);
                for (BGSIZE i = 0; i < synapse_counts; i++)
                    synapseDevice->postSpikeQueue->addAnEvent(incomingMap_begin[i], synapseDevice->total_delayPost[iSyn], iStepOffset);
                break;
            }
        case classAllSpikingSynapses:
        case classAllDSSynapses:
            break;
        default:
            assert(false);
        }
    } else {
		BGFLOAT r_sp += allNeuronsDevice->I0[idx]; // add IO
		allNeuronsDevice->hasFired[idx] = false;

        // Random number alg. goes here
        r_sp += (randNoise[idx] * allNeuronsDevice->Inoise[idx]); // add cheap noise
        vm = allNeuronsDevice->C1[idx] * vm + allNeuronsDevice->C2[idx] * r_sp; // decay Vm and add inputs
    }
}

/*
 *  CUDA code for advancing izhikevich neurons
 *
 *  @param[in] totalNeurons          Number of neurons.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] randNoise             Pointer to device random noise array.
 *  @param[in] allNeuronsDevice      Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesDevice     Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses 
 *                                   that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 */
__global__ void advanceIZHNeuronsDevice(const int totalNeurons, const int maxSpikes, 
                                        const BGFLOAT deltaT, const uint64_t simulationStep, const float* randNoise, 
                                        AllIZHNeuronsDeviceProperties* allNeuronsDevice, 
                                        AllSpikingSynapsesDeviceProperties* allSynapsesDevice, 
                                        SynapseIndexMap* synapseIndexMapDevice, 
                                        const bool fAllowBackPropagation, const int iStepOffset) {
    // determine which neuron this thread is processing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNeurons) return;

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
        allNeuronsDevice->spike_history[idx][idxSp] = simulationStep + iStepOffset;
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
            OUTGOING_SYNAPSE_INDEX_TYPE* outgoingMap_begin = &(synapseIndexMapDevice->outgoingSynapseIndexMap[beginIndex]);
                   
            // for each synapse, let them know we have fired
            for (BGSIZE i = 0; i < synapse_counts; i++) {
                OUTGOING_SYNAPSE_INDEX_TYPE idx = outgoingMap_begin[i];
                // outgoing synapse index consists of cluster index + synapse index
                CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIndex(idx);
                BGSIZE iSyn = SynapseIndexMap::getSynapseIndex(idx);
                allSynapsesDevice->preSpikeQueue->addAnEvent(iSyn, iCluster, iStepOffset);
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
                AllSTDPSynapsesDeviceProperties * synapseDevice = static_cast<AllSTDPSynapsesDeviceProperties *>(allSynapsesDevice);
                for (BGSIZE i = 0; i < synapse_counts; i++) {
                    synapseDevice->postSpikeQueue->addAnEvent(incomingMap_begin[i], synapseDevice->total_delayPost[iSyn], iStepOffset);
                } // end for
                break;
                    
            case classAllSpikingSynapses:
            case classAllDSSynapses:
                //for (BGSIZE i = 0; i < synapse_counts; i++) {
                //    postSpikingSynapsesSpikeHitDevice(incomingMap_begin[i], allSynapsesDevice, iStepOffset);
                //} // end for
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
