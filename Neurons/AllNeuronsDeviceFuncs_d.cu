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

__device__ void fireLIFDevice( const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, AllIFNeuronsProps* allNeuronsProps, int iStepOffset )
{
        int &nStepsInRefr = allNeuronsProps->nStepsInRefr[index];
        BGFLOAT &Trefract = allNeuronsProps->Trefract[index];
        int& spikeCount = allNeuronsProps->spikeCount[index];
        int& spikeCountOffset = allNeuronsProps->spikeCountOffset[index];
        bool &hasFired = allNeuronsProps->hasFired[index];
        BGFLOAT &Vm = allNeuronsProps->Vm[index];
        BGFLOAT &Vreset = allNeuronsProps->Vreset[index];

        // Note that the neuron has fired!
        hasFired = true;

        // record spike time
        int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
        allNeuronsProps->spike_history[index][idxSp] = simulationStep + iStepOffset;
        spikeCount++;

        DEBUG_SYNAPSE(
            printf("advanceLIFNeuronsDevice\n");
            printf("          index: %d\n", index);
            printf("          simulationStep: %d\n\n", simulationStep + iStepOffset);
        );

        // calculate the number of steps in the absolute refractory period
        nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

        // reset to 'Vreset'
        Vm = Vreset;
}

__device__ void advanceLIFNeuronDevice( const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIFNeuronsProps* allNeuronsProps, int iStepOffset )
{
        BGFLOAT &Vm = allNeuronsProps->Vm[index];
        BGFLOAT &Vthresh = allNeuronsProps->Vthresh[index];
        BGFLOAT &summationPoint = allNeuronsProps->summation_map[index];
        BGFLOAT &I0 = allNeuronsProps->I0[index];
        BGFLOAT &Inoise = allNeuronsProps->Inoise[index];
        BGFLOAT &C1 = allNeuronsProps->C1[index];
        BGFLOAT &C2 = allNeuronsProps->C2[index];
        int &nStepsInRefr = allNeuronsProps->nStepsInRefr[index];
        bool &hasFired = allNeuronsProps->hasFired[index];

        hasFired = false;

        if ( nStepsInRefr > 0 ) { // is neuron refractory?
                --nStepsInRefr;
        } else if ( Vm >= Vthresh ) { // should it fire?
                fireLIFDevice(index, maxSpikes, deltaT, simulationStep, allNeuronsProps, iStepOffset);
        } else {
                summationPoint += I0; // add IO

                // Random number alg. goes here
                summationPoint += (randNoise[index] * Inoise); // add cheap noise
                Vm = C1 * Vm + C2 * ( summationPoint ); // decay Vm and add inputs
        }

        // clear synaptic input for next time step
        summationPoint = 0;
}

__device__ void fireIZHDevice( const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, AllIZHNeuronsProps* allNeuronsProps, int iStepOffset )
{
        int &nStepsInRefr = allNeuronsProps->nStepsInRefr[index];
        BGFLOAT &Trefract = allNeuronsProps->Trefract[index];
        int &spikeCount = allNeuronsProps->spikeCount[index];
        int &spikeCountOffset = allNeuronsProps->spikeCountOffset[index];
        bool &hasFired = allNeuronsProps->hasFired[index];
        BGFLOAT &Vm = allNeuronsProps->Vm[index];
        BGFLOAT &u = allNeuronsProps->u[index];
        BGFLOAT &Cconst = allNeuronsProps->Cconst[index];
        BGFLOAT &Dconst = allNeuronsProps->Dconst[index];

        // Note that the neuron has fired!
        hasFired = true;

        // record spike time
        int idxSp = (spikeCount + spikeCountOffset) % maxSpikes;
        allNeuronsProps->spike_history[index][idxSp] = simulationStep + iStepOffset;
        spikeCount++;

        // calculate the number of steps in the absolute refractory period
        nStepsInRefr = static_cast<int> ( Trefract / deltaT + 0.5 );

        // reset to 'Vreset'
        Vm = Cconst * 0.001;
        u = u + Dconst;
}

__device__ void advanceIZHNeuronDevice( const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeuronsProps* allNeuronsProps, int iStepOffset )
{
        BGFLOAT &Vm = allNeuronsProps->Vm[index];
        BGFLOAT &Vthresh = allNeuronsProps->Vthresh[index];
        BGFLOAT &summationPoint = allNeuronsProps->summation_map[index];
        BGFLOAT &I0 = allNeuronsProps->I0[index];
        BGFLOAT &Inoise = allNeuronsProps->Inoise[index];
        BGFLOAT &C2 = allNeuronsProps->C2[index];
        BGFLOAT &C3 = allNeuronsProps->C3[index];
        int &nStepsInRefr = allNeuronsProps->nStepsInRefr[index];
        BGFLOAT &a = allNeuronsProps->Aconst[index];
        BGFLOAT &b = allNeuronsProps->Bconst[index];
        BGFLOAT &u = allNeuronsProps->u[index];
        bool &hasFired = allNeuronsProps->hasFired[index];

        hasFired = false;

        if ( nStepsInRefr > 0 ) { // is neuron refractory?
                --nStepsInRefr;
        } else if ( Vm >= Vthresh ) { // should it fire?
                fireIZHDevice( index, maxSpikes, deltaT, simulationStep, allNeuronsProps, iStepOffset );
        } else {
                summationPoint += I0; // add IO

                // Random number alg. goes here
                summationPoint += (randNoise[index] * Inoise); // add cheap noise

                BGFLOAT Vint = Vm * 1000;

                // Izhikevich model integration step
                BGFLOAT Vb = Vint + C3 * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
                u = u + C3 * a * (b * Vint - u);

                Vm = Vb * 0.001 + C2 * summationPoint;  // add inputs
        }

        // clear synaptic input for next time step
        summationPoint = 0;
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

        bool &hasFired = allNeuronsProps->hasFired[idx];

        advanceLIFNeuronDevice( idx, maxSpikes, deltaT, simulationStep, randNoise, allNeuronsProps, iStepOffset );

        // notify outgoing synapses of spike
        if (hasFired) {
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
        }
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

        bool &hasFired = allNeuronsProps->hasFired[idx];

        advanceIZHNeuronDevice( idx, maxSpikes, deltaT, simulationStep, randNoise, allNeuronsProps, iStepOffset );

                // notify outgoing synapses of spike
        if (hasFired) {
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
        }
}
