/*
 * AllIZHNeurons.cu
 *
 */

#include "AllSpikingSynapses.h"
#include "AllIZHNeurons.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	allocDeviceStruct( allNeurons, sim_info );

        HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIZHNeurons ) ) );
        HANDLE_ERROR( cudaMemcpy ( *allNeuronsDevice, &allNeurons, sizeof( AllIZHNeurons ), cudaMemcpyHostToDevice ) );
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the allIFNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
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

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );

	deleteDeviceStruct( allNeurons, sim_info );

	HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the allIFNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::deleteDeviceStruct( AllIZHNeurons& allNeurons, const SimulationInfo *sim_info ) {
	HANDLE_ERROR( cudaFree( allNeurons.Aconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Bconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Cconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.Dconst ) );
	HANDLE_ERROR( cudaFree( allNeurons.u ) );
	HANDLE_ERROR( cudaFree( allNeurons.C3 ) );

	AllIFNeurons::deleteDeviceStruct( allNeurons, sim_info );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info ) { 
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
	copyHostToDevice( allNeurons, sim_info );
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeurons         Reference to the allIFNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
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

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIZHNeurons allNeurons;

	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
	copyDeviceToHost( allNeurons, sim_info );
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeurons         Reference to the allIFNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
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

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons, sim_info );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons, sim_info );
}

/*
 *  Clear the spike counts out of all neurons.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIZHNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeurons, sim_info );
}

/*
 *  Notify outgoing synapses if neuron has fired.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void AllIZHNeurons::advanceNeurons( IAllSynapses &synapses, IAllNeurons* allNeuronsDevice, IAllSynapses* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice)
{
    int neuron_count = sim_info->totalNeurons;
    int maxSpikes = (int)((sim_info->epochDuration * sim_info->maxFiringRate));

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceIZHNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, sim_info->maxSynapsesPerNeuron, maxSpikes, sim_info->deltaT, g_simulationStep, randNoise, (AllIZHNeurons *)allNeuronsDevice, (AllSpikingSynapses*)allSynapsesDevice, synapseIndexMapDevice, (void (*)(const uint32_t, AllSpikingSynapses*))m_fpPreSpikeHit_h, (void (*)(const uint32_t, AllSpikingSynapses*))m_fpPostSpikeHit_h, m_fAllowBackPropagation );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

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
 *  @param[in] fpPreSpikeHit         Pointer to the device function preSpikeHit() function.
 *  @param[in] fpPostSpikeHit        Pointer to the device function postSpikeHit() function.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 */
__global__ void advanceIZHNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, AllIZHNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, SynapseIndexMap* synapseIndexMapDevice, void (*fpPreSpikeHit)(const uint32_t, AllSpikingSynapses*), void (*fpPostSpikeHit)(const uint32_t, AllSpikingSynapses*), bool fAllowBackPropagation ) {
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

                //notify incomming synapses of spike
                size_t synapse_counts = allSynapsesDevice->synapse_counts[idx];
                uint32_t synapse_notified = 0;
                if(fAllowBackPropagation && synapse_counts != 0){
                   for(uint32_t synapse_index = maxSynapses * idx ; synapse_notified < synapse_counts; synapse_index++){
                      if (allSynapsesDevice->in_use[synapse_index] == true) {
                         fpPreSpikeHit(synapse_index, allSynapsesDevice); 
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

                BGFLOAT Vint = r_vm * 1000;

                // Izhikevich model integration step
                BGFLOAT Vb = Vint + allNeuronsDevice->C3[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
                u = r_u + allNeuronsDevice->C3[idx] * r_a * (r_b * Vint - r_u);

                vm = Vb * 0.001 + allNeuronsDevice->C2[idx] * r_sp;  // add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
}

