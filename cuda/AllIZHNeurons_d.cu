/*
 * AllIZHNeurons.cu
 *
 */

#include "AllIZHNeurons.h"
#include "AllSpikingSynapses.h"
#include "Book.h"

void AllIZHNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	allocDeviceStruct( allNeurons, sim_info );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIZHNeurons ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIZHNeurons ), cudaMemcpyHostToDevice ) );
}

void AllIZHNeurons::allocDeviceStruct( AllIZHNeurons &allNeurons, SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;

	AllIFNeurons::allocDeviceStruct( allNeurons, sim_info );
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Aconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Bconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Cconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Dconst, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.u, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C3, count * sizeof( BGFLOAT ) ) );
}

void AllIZHNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons, sim_info );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

void AllIZHNeurons::deleteDeviceStruct( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info ) {
	HANDLE_ERROR( cudaFree( allNeurons.Aconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Bconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Cconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Dconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.u ) );
	HANDLE_ERROR( cudaFree( allNeurons.C3 ) );

	AllIFNeurons::deleteDeviceStruct( allNeurons, sim_info );
}

void AllIZHNeurons::copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) { 
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons, sim_info );
}

void AllIZHNeurons::copyHostToDevice( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info ) { 
	int count = sim_info->totalNeurons;

	AllIFNeurons::copyHostToDevice( allNeurons, sim_info );

	HANDLE_ERROR( cudaMemcpy ( allNeurons.Aconst, Aconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Bconst, Bconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Cconst, Cconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.Dconst, Dconst, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.u, u, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( allNeurons.C3, C3, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
}

void AllIZHNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons, sim_info );
}

void AllIZHNeurons::copyDeviceToHost( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;

	AllIFNeurons::copyDeviceToHost( allNeurons, sim_info );

	HANDLE_ERROR( cudaMemcpy ( Aconst, allNeurons.Aconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Bconst, allNeurons.Bconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cconst, allNeurons.Cconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Dconst, allNeurons.Dconst, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( u, allNeurons.u, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C3, allNeurons.C3, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
}

/**
 *  Get spike history in AllIZHNeurons struct on device memory.
 *  @param  allNeuronsDevice      Reference to the allNeurons struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons, sim_info );
}

/**
 *  Get spikeCount in AllIZHNeurons struct on device memory.
 *  @param  allNeuronsDevice      Reference to the allNeurons struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons, sim_info );
}

/** 
*  Clear the spike counts out of all Neurons.
 *  @param  allNeuronsDevice      Reference to the allNeurons struct on device memory.
 *  @param  sim_info    SimulationInfo to refer from.
*/
void AllIZHNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeurons, sim_info );
}

//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, void (*fpPreSpikeHit)(const uint32_t, AllSpikingSynapses*), void (*fpPostSpikeHit)(const uint32_t, AllSpikingSynapses*), bool fAllowBackPropagation );

/**
 *  Notify outgoing synapses if neuron has fired.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::advanceNeurons( AllSynapses &synapses, AllNeurons* allNeuronsDevice, AllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice)
{
    int neuron_count = sim_info->totalNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    AllSpikingSynapses &spSynapses = dynamic_cast<AllSpikingSynapses&>(synapses);
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    bool fAllowBackPropagation = spSynapses.allowBackPropagation();
    unsigned long long fpPreSpikeHit_h, fpPostSpikeHit_h;
    spSynapses.getFpPrePostSpikeHit(fpPreSpikeHit_h, fpPostSpikeHit_h);

    advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIZHNeurons *)allNeuronsDevice, (AllSpikingSynapses*)allSynapsesDevice, synapseIndexMapDevice, (void (*)(const uint32_t, AllSpikingSynapses*))fpPreSpikeHit_h, (void (*)(const uint32_t, AllSpikingSynapses*))fpPostSpikeHit_h, fAllowBackPropagation );
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
__global__ void advanceNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, void (*fpPreSpikeHit)(const uint32_t, AllSpikingSynapses*), void (*fpPostSpikeHit)(const uint32_t, AllSpikingSynapses*), bool fAllowBackPropagation ) {
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

                BGFLOAT Vint = r_vm * 1000;

                // Izhikevich model integration step
                BGFLOAT Vb = Vint + allNeuronsDevice->C3[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
                u = r_u + allNeuronsDevice->C3[idx] * r_a * (r_b * Vint - r_u);

                vm = Vb * 0.001 + allNeuronsDevice->C2[idx] * r_sp;  // add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
}

