/*
 * AllIFNeurons_d.cu
 *
 */

#include "AllIFNeurons.h"
#include "Book.h"

/*
 *  Allocate GPU memories to store all neurons' states,
 *  and copy them from host to GPU memory.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info ) {
	AllIFNeurons allNeurons;
   for(int i =0; i < sim_info->numGPU; i++){
      cudaSetDevice(i);
      allocDeviceStruct( allNeurons, &sim_info->individualGPUInfo[i] );
      HANDLE_ERROR( cudaMalloc( allNeuronsDevice, sizeof( AllIFNeurons ) ) );
      HANDLE_ERROR( cudaMemcpy ( allNeuronsDevice[i], &allNeurons, sizeof( AllIFNeurons ), cudaMemcpyHostToDevice ) );\
   }
}

/*
 *  Allocate GPU memories to store all neurons' states.
 *  (Helper function of allocNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the allIFNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::allocDeviceStruct( AllIFNeurons &allNeurons, SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;
	int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
 
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C1, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.C2, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Cm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.I0, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Iinject, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Inoise, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Isyn, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Rm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Tau, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Trefract, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vinit, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vreset, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vrest, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.Vthresh, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.hasFired, count * sizeof( bool ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.nStepsInRefr, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spikeCount, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spikeCountOffset, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.summation_map, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &allNeurons.spike_history, count * sizeof( uint64_t* ) ) );
	
	uint64_t* pSpikeHistory[count]; //make a "count" length array of uint64_t pointers
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &pSpikeHistory[i], max_spikes * sizeof( uint64_t ) ) );
	}
	HANDLE_ERROR( cudaMemcpy ( allNeurons.spike_history, pSpikeHistory,
		count * sizeof( uint64_t* ), cudaMemcpyHostToDevice ) );

	// get device summation point address and set it to sim info
	sim_info->pSummationMap = allNeurons.summation_map;
}

/*
 *  Delete GPU memories.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::deleteNeuronDeviceStruct( void* allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIFNeurons allNeurons;
   
   for(int i =0; i < sim_info->numGPU; i++){
      cudaSetDevice(i);
	   HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
	   deleteDeviceStruct( allNeurons, sim_info );
      HANDLE_ERROR( cudaFree( allNeuronsDevice ) );
   }
}

/*
 *  Delete GPU memories.
 *  (Helper function of deleteNeuronDeviceStruct)
 *
 *  @param  allNeurons         Reference to the allIFNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::deleteDeviceStruct( AllIFNeurons& allNeurons, const SimulationInfo *sim_info ) {
	int count = sim_info->totalNeurons;

	uint64_t* pSpikeHistory[count];
	HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history,
		count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR( cudaFree( pSpikeHistory[i] ) );
	}

	HANDLE_ERROR( cudaFree( allNeurons.C1 ) );
	HANDLE_ERROR( cudaFree( allNeurons.C2 ) );
	HANDLE_ERROR( cudaFree( allNeurons.Cm ) );
	HANDLE_ERROR( cudaFree( allNeurons.I0 ) );
	HANDLE_ERROR( cudaFree( allNeurons.Iinject ) );
	HANDLE_ERROR( cudaFree( allNeurons.Inoise ) );
	HANDLE_ERROR( cudaFree( allNeurons.Isyn ) );
	HANDLE_ERROR( cudaFree( allNeurons.Rm ) );
	HANDLE_ERROR( cudaFree( allNeurons.Tau ) );
	HANDLE_ERROR( cudaFree( allNeurons.Trefract ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vinit ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vm ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vreset ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vrest ) );
	HANDLE_ERROR( cudaFree( allNeurons.Vthresh ) );
	HANDLE_ERROR( cudaFree( allNeurons.hasFired ) );
	HANDLE_ERROR( cudaFree( allNeurons.nStepsInRefr ) );
	HANDLE_ERROR( cudaFree( allNeurons.spikeCount ) );
	HANDLE_ERROR( cudaFree( allNeurons.spikeCountOffset ) );
	HANDLE_ERROR( cudaFree( allNeurons.summation_map ) );
	HANDLE_ERROR( cudaFree( allNeurons.spike_history ) );
}

/*
 *  Copy all neurons' data from host to device.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::copyNeuronHostToDevice( void** allNeuronsDevice, const SimulationInfo *sim_info ) { 
	AllIFNeurons allNeurons;
   int offsetFromFirstNeuron = 0;
   for(int i =0; i < sim_info->numGPU; i++){
      cudaSetDevice(i);
      //copy the base addresses for all the arrays on the device
      HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice[i], sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
      
      copyHostToDevice( allNeurons, &sim_info->individualGPUInfo[i], offsetFromFirstNeuron );
      offsetFromFirstNeuron += sim_info->individualGPUInfo[i].totalNeurons;
   }
}

/*
 *  Copy all neurons' data from host to device.
 *  (Helper function of copyNeuronHostToDevice)
 *
 *  @param  allNeurons            Reference to the allIFNeurons struct.
 *  @param  sim_info              SimulationInfo to refer from.
 *  @param  offsetFromFirstNeuron 
 */
void AllIFNeurons::copyHostToDevice( AllIFNeurons& allNeurons, const SimulationInfo *sim_info, const int offsetFromFirstNeuron ) { 

   int count = sim_info->totalNeurons;

   HANDLE_ERROR( cudaMemcpy ( allNeurons.C1, C1+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.C2, C2+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Cm, Cm+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.I0, I0+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Iinject, Iinject+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Inoise, Inoise+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Isyn, Isyn+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Rm, Rm+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Tau, Tau+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Trefract, Trefract+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Vinit, Vinit+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Vm, Vm+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Vreset, Vreset+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Vrest, Vrest+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.Vthresh, Vthresh+offsetFromFirstNeuron, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.hasFired, hasFired+offsetFromFirstNeuron, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.nStepsInRefr, nStepsInRefr+offsetFromFirstNeuron, count * sizeof( int ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeCount, spikeCount+offsetFromFirstNeuron, count * sizeof( int ), cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy ( allNeurons.spikeCountOffset, spikeCountOffset+offsetFromFirstNeuron, count * sizeof( int ), cudaMemcpyHostToDevice ) );

   int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
   uint64_t* pSpikeHistory[count];
   HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
   for (int i = 0; i < count; i++) {
          HANDLE_ERROR( cudaMemcpy ( pSpikeHistory[i], spike_history[i + offsetFromFirstNeuron], max_spikes * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
   }
}

/*
 *  Copy all neurons' data from device to host.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceToHost( void** allNeuronsDevice, const SimulationInfo *sim_info ) {
	AllIFNeurons allNeurons;

   int offsetFromFirstNeuron = 0;
   for(int i =0; i < sim_info->numGPU; i++){
      cudaSetDevice(i);
      //get pointers to arrays on device
      HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice[i], sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
      copyDeviceToHost( allNeurons, &sim_info->individualGPUInfo[i], offsetFromFirstNeuron );
      offsetFromFirstNeuron += sim_info->individualGPUInfo[i].totalNeurons;
   }
}

/*
 *  Copy all neurons' data from device to host.
 *  (Helper function of copyNeuronDeviceToHost)
 *
 *  @param  allNeurons         Reference to the allIFNeurons struct.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::copyDeviceToHost( AllIFNeurons& allNeurons, const SimulationInfo *sim_info, const int offsetFromFirstNeuron ) {
	int count = sim_info->totalNeurons;

	HANDLE_ERROR( cudaMemcpy ( C1+offsetFromFirstNeuron, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( C2+offsetFromFirstNeuron, allNeurons.C2, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Cm+offsetFromFirstNeuron, allNeurons.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( I0+offsetFromFirstNeuron, allNeurons.I0, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Iinject+offsetFromFirstNeuron, allNeurons.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Inoise+offsetFromFirstNeuron, allNeurons.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Isyn+offsetFromFirstNeuron, allNeurons.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Rm+offsetFromFirstNeuron, allNeurons.Rm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Tau+offsetFromFirstNeuron, allNeurons.Tau, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Trefract+offsetFromFirstNeuron, allNeurons.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vinit+offsetFromFirstNeuron, allNeurons.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vm+offsetFromFirstNeuron, allNeurons.Vm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vreset+offsetFromFirstNeuron, allNeurons.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vrest+offsetFromFirstNeuron, allNeurons.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( Vthresh+offsetFromFirstNeuron, allNeurons.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( hasFired+offsetFromFirstNeuron, allNeurons.hasFired, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( nStepsInRefr+offsetFromFirstNeuron, allNeurons.nStepsInRefr, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCount+offsetFromFirstNeuron, allNeurons.spikeCount, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCountOffset+offsetFromFirstNeuron, allNeurons.spikeCountOffset, count * sizeof( int ), cudaMemcpyDeviceToHost ) );

   int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
   uint64_t* pSpikeHistory[count];
   HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history, count * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );
   for (int i = 0; i < count; i++) {
      HANDLE_ERROR( cudaMemcpy ( spike_history[i+offsetFromFirstNeuron], pSpikeHistory[i], max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
   }
}

/*
 *  Copy spike history data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info ) 
{        
        AllIFNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );        
        AllSpikingNeurons::copyDeviceSpikeHistoryToHost( allNeurons, sim_info );
}

/*
 *  Copy spike counts data stored in device memory to host.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIFNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::copyDeviceSpikeCountsToHost( allNeurons, sim_info );
}

/*
 *  Clear the spike counts out of all neurons.
 *
 *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
 *  @param  sim_info           SimulationInfo to refer from.
 */
void AllIFNeurons::clearNeuronSpikeCounts( void* allNeuronsDevice, const SimulationInfo *sim_info )
{
        AllIFNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
        AllSpikingNeurons::clearDeviceSpikeCounts( allNeurons, sim_info );
}

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
void AllIFNeurons::advanceNeurons( IAllSynapses &synapses, IAllNeurons** allNeuronsDevice, IAllSynapses** allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap** synapseIndexMapDevice )
{
}
