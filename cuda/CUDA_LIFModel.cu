/** \file CUDA_LIFModel.cu (Paul Bunn)
 ** 
 ** \derived from GpuSim_struct.cu
 **
 ** \authors Fumitaka Kawasaki
 **
 ** \brief Functions that perform the GPU version of simulation.
 **/

#define _CUDA_LIFModel
#include "BGTypes.h"
#include "MersenneTwisterGPU.h"
#include "../tinyxml/tinyxml.h"

#include "ParseParamError.h"
#include "Util.h"
#include "Global.h"
#include "Book.h"
#include "CUDA_LIFModel.h"


CUDA_LIFModel::CUDA_LIFModel()
{
}

CUDA_LIFModel::~CUDA_LIFModel()
{
#ifdef STORE_SPIKEHISTORY
    delete[] spikeArray;
#endif // STORE_SPIKEHISTORY
	delete[] m_conns->spikeCounts;
	m_conns->spikeCounts = NULL;
	deleteDeviceStruct();
}

bool CUDA_LIFModel::initializeModel(SimulationInfo *sim_info, AllNeurons& neurons, AllSynapses& synapses)
{
	LifNeuron_struct neuron_st;
	LifSynapse_struct synapse_st;

	// copy synapse and neuron maps into arrays
	dataToCStructs(sim_info, neurons, synapses, neuron_st, synapse_st);

	uint32_t neuron_count = sim_info->cNeurons;
	delete[] m_conns->spikeCounts;
	m_conns->spikeCounts = new uint32_t[neuron_count]();

#ifdef STORE_SPIKEHISTORY
	uint32_t maxSpikes = static_cast<uint32_t> (sim_info->epochDuration * sim_info->maxFiringRate);
    spikeArray = new uint64_t[maxSpikes * neuron_count]();

	// allocate device memory
	allocDeviceStruct(sim_info, neuron_st, synapse_st, neurons, synapses, sim_info->maxSynapsesPerNeuron, maxSpikes);
#else
	// allocate device memory
	allocDeviceStruct(sim_info, neuron_st, synapse_st, neurons, synapses, sim_info->maxSynapsesPerNeuron);
#endif // STORE_SPIKEHISTORY

	//initialize Mersenne Twister
	//assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
	uint32_t rng_blocks = 25; //# of blocks the kernel will use
	uint32_t rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
	uint32_t rng_mt_rng_count = sim_info->cNeurons/rng_nPerRng; //# of threads to generate for neuron_count rand #s
	uint32_t rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
	initMTGPU(777, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

	// delete the arrays
	deleteNeuronStruct(neuron_st);
	deleteSynapseStruct(synapse_st);
	return true;
}

/**
 * Copy synapse and neuron C++ objects into C structs
 * @param psi simulation info
 * @param [out] synapse_st object stores synapse data in arrays
 * @param [out] neuron_st object stores neuron data in arrays
 * @param [out] neuron_count
 * @param [out] synapse_count
 */ 
void CUDA_LIFModel::dataToCStructs(SimulationInfo *psi, AllNeurons& neurons, AllSynapses& synapses, LifNeuron_struct &neuron_st, LifSynapse_struct &synapse_st) 
{
	// Allocate memory
	uint32_t neuron_count = psi->cNeurons;
	allocNeuronStruct(neuron_st, neuron_count);
	allocSynapseStruct(synapse_st, neuron_count * psi->maxSynapsesPerNeuron);

	// Copy memory
	for (uint32_t i = 0; i < neuron_count; i++)
	{
		copyNeuronToStruct(neurons, neuron_st, i);
		neuron_st.synapseCount[i] = synapses.synapse_counts[i];
		assert(neuron_st.synapseCount[i] <= psi->maxSynapsesPerNeuron);
		neuron_st.outgoingSynapse_begin[i] = i * psi->maxSynapsesPerNeuron;

		for (uint32_t j = 0; j < synapses.synapse_counts[i]; j++)
		{
			copySynapseToStruct(psi->cNeurons, synapses, i, synapse_st, j);
		}
	}
}

/**
 * @param[in] neuron_count	Number of neurons.
 * @param[out] spikeCounts	Array to store spike counts for neurons. 
 */
void CUDA_LIFModel::readSpikesFromDevice(uint32_t neuron_count, uint32_t *spikeCounts)
{
	LifNeuron_struct neuron;

    HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCounts, neuron.spikeCount, neuron_count * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
}

/**
 * @param[in] neuron_count	Number of neurons.
 */
void CUDA_LIFModel::clearSpikesFromDevice(uint32_t neuron_count)
{
	LifNeuron_struct neuron;
    HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );
	HANDLE_ERROR( cudaMemset( neuron.spikeCount, 0, neuron_count * sizeof( uint32_t ) ) );
}

void CUDA_LIFModel::advance(AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info)
{
#ifdef STORE_SPIKEHISTORY
	uint32_t maxSpikes = static_cast<uint32_t> (sim_info->epochDuration * sim_info->maxFiringRate);
#endif // STORE_SPIKEHISTORY

#ifdef STORE_SPIKEHISTORY
	advanceGPU(sim_info, neurons, synapses, sim_info->maxSynapsesPerNeuron, spikeArray, maxSpikes);
#else
	advanceGPU(sim_info, neurons, synapses, sim_info->maxSynapsesPerNeuron);
#endif // STORE_SPIKEHISTORY

#ifdef STORE_SPIKEHISTORY
	uint32_t neuron_count = sim_info->cNeurons;

	// record spike time      neurons.spike_history[index][neurons.totalSpikeCount[index]] = g_simulationStep;

	readSpikesFromDevice(neuron_count, m_conns->spikeCounts);
	for (uint32_t i = 0; i < neuron_count; i++) {
		if (m_conns->spikeCounts[i] > 0) {
			assert(m_conns->spikeCounts[i] + neurons.totalSpikeCount[i] < maxSpikes);
			for(uint32_t j = 0; j < m_conns->spikeCounts[i]; j++) {
				neurons.spike_history[i][neurons.totalSpikeCount[i]] = spikeArray[i * maxSpikes + j];
				neurons.totalSpikeCount[i]++;
				neurons.spikeCount[i]++;
			}
		}
	}
	clearSpikesFromDevice(neuron_count);
#endif // STORE_SPIKEHISTORY
}

/**
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void CUDA_LIFModel::updateWeights(const uint32_t neuron_count, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info)
{
	uint32_t width = sim_info->width;
	TIMEFLOAT deltaT = sim_info->deltaT;

    // CUDA parameters
    const uint32_t threadsPerBlock = 256;
    uint32_t blocksPerGrid;

	// allocate memories
	size_t W_d_size = neuron_count * neuron_count * sizeof (float);
	float* W_h = new BGFLOAT[W_d_size]; // No need to initialize -- will be filled in immediately below
	float* W_d;
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

	// copy weight data to the device memory
	for ( uint32_t i = 0 ; i < neuron_count; i++ )
		for ( uint32_t j = 0; j < neuron_count; j++ )
			W_h[i * neuron_count + j] = m_conns->W(i, j);

	HANDLE_ERROR( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

	blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
	updateNetworkDevice <<< blocksPerGrid, threadsPerBlock >>> ( summationPoint_d, rgNeuronTypeMap_d, neuron_count, width, deltaT, W_d, sim_info->maxSynapsesPerNeuron );

	// free memories
	HANDLE_ERROR( cudaFree( W_d ) );
	delete[] W_h;

	// create synapse inverse map
	createSynapseImap( sim_info, sim_info->maxSynapsesPerNeuron );
}

void allocNeuronStruct_d( uint32_t count ) {
	LifNeuron_struct neuron;

	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.deltaT, count * sizeof( TIMEFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.summationPoint, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Cm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Rm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vthresh, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vrest, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vreset, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vinit, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Trefract, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Inoise, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.randNoise, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Iinject, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Isyn, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.nStepsInRefr, count * sizeof( uint32_t) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.C1, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.C2, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.I0, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Vm, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.hasFired, count * sizeof( char ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.Tau, count * sizeof( BGFLOAT ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.spikeCount, count * sizeof( uint32_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.outgoingSynapse_begin, count * sizeof( uint32_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.synapseCount, count * sizeof( uint32_t ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron.incomingSynapse_begin, count * sizeof( uint32_t ) ) );
	
	HANDLE_ERROR( cudaMemcpyToSymbol ( neuron_st_d, &neuron, sizeof( LifNeuron_struct ) ) );
}

void deleteNeuronStruct_d(  ) {
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );

	HANDLE_ERROR( cudaFree( neuron.deltaT ) );
	HANDLE_ERROR( cudaFree( neuron.summationPoint ) );
	HANDLE_ERROR( cudaFree( neuron.Cm ) );
	HANDLE_ERROR( cudaFree( neuron.Rm ) );
	HANDLE_ERROR( cudaFree( neuron.Vthresh ) );
	HANDLE_ERROR( cudaFree( neuron.Vrest ) );
	HANDLE_ERROR( cudaFree( neuron.Vreset ) );
	HANDLE_ERROR( cudaFree( neuron.Vinit ) );
	HANDLE_ERROR( cudaFree( neuron.Trefract ) );
	HANDLE_ERROR( cudaFree( neuron.Inoise ) );
	HANDLE_ERROR( cudaFree( neuron.randNoise ) );
	HANDLE_ERROR( cudaFree( neuron.Iinject ) );
	HANDLE_ERROR( cudaFree( neuron.Isyn ) );
	HANDLE_ERROR( cudaFree( neuron.nStepsInRefr ) );
	HANDLE_ERROR( cudaFree( neuron.C1 ) );
	HANDLE_ERROR( cudaFree( neuron.C2 ) );
	HANDLE_ERROR( cudaFree( neuron.I0 ) );
	HANDLE_ERROR( cudaFree( neuron.Vm ) );
	HANDLE_ERROR( cudaFree( neuron.hasFired ) );
	HANDLE_ERROR( cudaFree( neuron.Tau ) );
	HANDLE_ERROR( cudaFree( neuron.spikeCount ) );
	HANDLE_ERROR( cudaFree( neuron.outgoingSynapse_begin ) );
	HANDLE_ERROR( cudaFree( neuron.synapseCount ) );
	HANDLE_ERROR( cudaFree( neuron.incomingSynapse_begin ) );
}

void copyNeuronHostToDevice( LifNeuron_struct& neuron_h, uint32_t count ) {
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );

	HANDLE_ERROR( cudaMemcpy ( neuron.deltaT, neuron_h.deltaT, count * sizeof( TIMEFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.summationPoint, neuron_h.summationPoint, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Cm, neuron_h.Cm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Rm, neuron_h.Rm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vthresh, neuron_h.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vrest, neuron_h.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vreset, neuron_h.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vinit, neuron_h.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Trefract, neuron_h.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Inoise, neuron_h.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Iinject, neuron_h.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Isyn, neuron_h.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.nStepsInRefr, neuron_h.nStepsInRefr, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.C1, neuron_h.C1, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.C2, neuron_h.C2, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.I0, neuron_h.I0, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Vm, neuron_h.Vm, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.hasFired, neuron_h.hasFired, count * sizeof( char ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.Tau, neuron_h.Tau, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.spikeCount, neuron_h.spikeCount, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.outgoingSynapse_begin, neuron_h.outgoingSynapse_begin, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.synapseCount, neuron_h.synapseCount, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.incomingSynapse_begin, neuron_h.incomingSynapse_begin, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
}

void copyNeuronDeviceToHost( LifNeuron_struct& neuron_h, uint32_t count ) {
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );

	HANDLE_ERROR( cudaMemcpy ( neuron_h.C1, neuron.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.C2, neuron.C2, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Cm, neuron.C1, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.I0, neuron.I0, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Iinject, neuron.Iinject, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Inoise, neuron.Inoise, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Isyn, neuron.Isyn, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Rm, neuron.Rm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Tau, neuron.Tau, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Trefract, neuron.Trefract, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vinit, neuron.Vinit, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vm, neuron.Vm, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vrest, neuron.Vrest, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vreset, neuron.Vreset, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.Vthresh, neuron.Vthresh, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.nStepsInRefr, neuron.nStepsInRefr, count * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.spikeCount, neuron.spikeCount, count * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( neuron_h.synapseCount, neuron.synapseCount, count * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
}

void allocSynapseStruct_d( uint32_t count ) {
	LifSynapse_struct synapse;

	if ( count > 0 ) {
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.inUse, count * sizeof( bool ) ) );
		HANDLE_ERROR( cudaMemset( synapse.inUse, 0, count * sizeof( bool ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.summationPoint, count * sizeof( PBGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.summationCoord, count * sizeof( Coordinate ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.synapseCoord, count * sizeof( Coordinate ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.deltaT, count * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.W, count * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.psr, count * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.decay, count * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.total_delay, count * sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.type, count * sizeof( synapseType ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.delayQueue, count * sizeof( uint32_t ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.tau, count * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.r, count * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.u, count * sizeof( BGFLOAT ) ) );
		HANDLE_ERROR( cudaMalloc( ( void ** ) &synapse.lastSpike, count * sizeof( uint64_t ) ) );

		HANDLE_ERROR( cudaMemcpyToSymbol ( synapse_st_d, &synapse, sizeof( LifSynapse_struct ) ) );
	}
}

void deleteSynapseStruct_d( ) {
	LifSynapse_struct synapse;

	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( LifSynapse_struct ) ) );

	HANDLE_ERROR( cudaFree( synapse.inUse ) );
	HANDLE_ERROR( cudaFree( synapse.summationPoint ) );
	HANDLE_ERROR( cudaFree( synapse.summationCoord ) );
	HANDLE_ERROR( cudaFree( synapse.synapseCoord ) );
	HANDLE_ERROR( cudaFree( synapse.deltaT ) );
	HANDLE_ERROR( cudaFree( synapse.W ) );
	HANDLE_ERROR( cudaFree( synapse.psr ) );
	HANDLE_ERROR( cudaFree( synapse.decay ) );
	HANDLE_ERROR( cudaFree( synapse.total_delay ) );
	HANDLE_ERROR( cudaFree( synapse.type ) );
	HANDLE_ERROR( cudaFree( synapse.delayQueue ) );
	HANDLE_ERROR( cudaFree( synapse.tau ) );
	HANDLE_ERROR( cudaFree( synapse.r ) );
	HANDLE_ERROR( cudaFree( synapse.u ) );
	HANDLE_ERROR( cudaFree( synapse.lastSpike ) );
}

void copySynapseHostToDevice( LifSynapse_struct& synapse_h, uint32_t count ) {
	// copy everything necessary
	LifSynapse_struct synapse;

	if ( count > 0 ) {
        	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( LifSynapse_struct ) ) );

		HANDLE_ERROR( cudaMemcpy ( synapse.inUse, synapse_h.inUse, count * sizeof( bool ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.summationCoord, synapse_h.summationCoord, count * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.synapseCoord, synapse_h.synapseCoord, count * sizeof( Coordinate ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.summationPoint, synapse_h.summationPoint, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.deltaT, synapse_h.deltaT, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.W, synapse_h.W, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.psr, synapse_h.psr, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.decay, synapse_h.decay, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.total_delay, synapse_h.total_delay, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.type, synapse_h.type, count * sizeof( synapseType ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.delayQueue, synapse_h.delayQueue, count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.r, synapse_h.r, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.u, synapse_h.u, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.tau, synapse_h.tau, count * sizeof( BGFLOAT ), cudaMemcpyHostToDevice ) );
		HANDLE_ERROR( cudaMemcpy ( synapse.lastSpike, synapse_h.lastSpike, count * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );
	}
}

void copySynapseDeviceToHost( LifSynapse_struct& synapse_h, uint32_t count ) {
	// copy everything necessary
	LifSynapse_struct synapse;

	if ( count > 0 ) {
        	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( LifSynapse_struct ) ) );

		HANDLE_ERROR( cudaMemcpy ( synapse_h.inUse, synapse.inUse, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.summationCoord, synapse.summationCoord, count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.synapseCoord, synapse.synapseCoord, count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.deltaT, synapse.deltaT, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.W, synapse.W, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.psr, synapse.psr, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.decay, synapse.decay, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.total_delay, synapse.total_delay, count * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.type, synapse.type, count * sizeof( synapseType ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.r, synapse.r, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.u, synapse.u, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.tau, synapse.tau, count * sizeof( BGFLOAT ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.delayQueue, synapse.delayQueue, count * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.lastSpike, synapse.lastSpike, count * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
	}
}

void copySynapseSumCoordDeviceToHost( LifSynapse_struct& synapse_h, uint32_t count ) {
	// copy everything necessary
	LifSynapse_struct synapse;

	if ( count > 0 ) {
        	HANDLE_ERROR( cudaMemcpyFromSymbol ( &synapse, synapse_st_d, sizeof( LifSynapse_struct ) ) );

		HANDLE_ERROR( cudaMemcpy ( synapse_h.inUse, synapse.inUse, count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy ( synapse_h.summationCoord, synapse.summationCoord, count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
	}
}

/**
 * @param[in] psi		Pointer to the simulation information.
 * @param[in] neuron_st		A leaky-integrate-and-fire (I&F) neuron structure.
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 * @param[in] maxSpikes		Maximum number of spikes per neuron per one epoch.
 */
void allocDeviceStruct(SimulationInfo * psi, 
		LifNeuron_struct& neuron_st,
		LifSynapse_struct& synapse_st,
		AllNeurons& neurons,
		AllSynapses& synapses,
#ifdef STORE_SPIKEHISTORY
        uint32_t maxSynapses,
		uint32_t maxSpikes 
#else
        uint32_t maxSynapses
#endif // STORE_SPIKEHISTORY
		)
{
	// Set device ID
	HANDLE_ERROR( cudaSetDevice( g_deviceId ) );

	// CUDA parameters
	const uint32_t threadsPerBlock = 256;
	uint32_t blocksPerGrid;

	// Allocate GPU device memory
	uint32_t neuron_count = psi->cNeurons;
	uint32_t synapse_count = neuron_count * maxSynapses;
	allocNeuronStruct_d( neuron_count );				// and allocate device memory for each member
	allocSynapseStruct_d( synapse_count );				// and allocate device memory for each member

#ifdef STORE_SPIKEHISTORY
	spikeHistory_d_size = neuron_count * maxSpikes * sizeof (uint64_t);		// size of spike history array
#endif // STORE_SPIKEHISTORY
	size_t summationPoint_d_size = neuron_count * sizeof (BGFLOAT);	// size of summation point
	size_t randNoise_d_size = neuron_count * sizeof (float);	// size of random noise array
	size_t rgNeuronTypeMap_d_size = neuron_count * sizeof(neuronType);

#ifdef STORE_SPIKEHISTORY
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &spikeHistory_d, spikeHistory_d_size ) );
#endif // STORE_SPIKEHISTORY
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &summationPoint_d, summationPoint_d_size ) );
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &rgNeuronTypeMap_d, rgNeuronTypeMap_d_size ) );

	// Copy host neuron and synapse arrays into GPU device
	copyNeuronHostToDevice( neuron_st, neuron_count );
	copySynapseHostToDevice( synapse_st, synapse_count );

	// Copy neuron type map into device memory
	HANDLE_ERROR( cudaMemcpy ( rgNeuronTypeMap_d,  neurons.neuron_type_map, rgNeuronTypeMap_d_size, cudaMemcpyHostToDevice ) );

	uint32_t width = psi->width;	
	blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
	calcOffsets<<< blocksPerGrid, threadsPerBlock >>>( neuron_count, summationPoint_d, width, randNoise_d );

	// create synapse inverse map
	createSynapseImap( psi, maxSynapses );
}

/**
 */
void deleteDeviceStruct(  )
{
	// Deallocate device memory
	deleteNeuronStruct_d(  );
	deleteSynapseStruct_d(  );
#ifdef STORE_SPIKEHISTORY
	HANDLE_ERROR( cudaFree( spikeHistory_d ) );
	spikeHistory_d_size = 0;
	spikeHistory_d = NULL;
#endif // STORE_SPIKEHISTORY
	HANDLE_ERROR( cudaFree( summationPoint_d ) );
	HANDLE_ERROR( cudaFree( randNoise_d ) );
	if ( inverseMap_d != NULL)
		HANDLE_ERROR( cudaFree( inverseMap_d ) );
	HANDLE_ERROR( cudaFree( rgNeuronTypeMap_d ) );
}

#ifdef STORE_SPIKEHISTORY
/**
 * @param[in] psi		Pointer to the simulation information. 
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 * @param[in] spikeArray	Array to save spike history for neurons. 
 * @param[in] maxSpikes		Maximum number of spikes per neuron per one epoch.
 */
void advanceGPU(SimulationInfo *psi, AllNeurons& neurons, AllSynapses& synapses, uint32_t maxSynapses, uint64_t* spikeArray, uint32_t maxSpikes )
#else
/**
 * @param[in] psi		Pointer to the simulation information. 
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
void advanceGPU(SimulationInfo *psi, AllNeurons& neurons, AllSynapses& synapses, uint32_t maxSynapses )
#endif
{
	BGFLOAT deltaT = psi->deltaT;
	uint32_t width = psi->width;
	uint32_t neuron_count = psi->cNeurons;
	uint32_t synapse_count = neuron_count * maxSynapses;

    // simulate to next growth cycle
    uint64_t endStep = g_simulationStep + static_cast<uint64_t>(psi->epochDuration / deltaT);
	
	DEBUG(cout << "Beginning GPU sim cycle, simTime = " << g_simulationStep * deltaT << ", endTime = " << endStep * deltaT << endl;)

	// CUDA parameters
	const uint32_t threadsPerBlock = 256;
	uint32_t blocksPerGrid;

	uint64_t count = 0;

#ifdef PERFORMANCE_METRICS
	cudaEvent_t start, stop;
	float time;
	t_gpu_rndGeneration = 0.0f;
	t_gpu_advanceNeurons = 0.0f;
	t_gpu_advanceSynapses = 0.0f;
	t_gpu_calcSummation = 0.0f;

	cudaEventCreate( &start );
	cudaEventCreate( &stop ); 
#endif // PERFORMANCE_METRICS
	while ( g_simulationStep < endStep )
	{
        	DEBUG( if (count %10000 == 0)
              	{
                  	cout << psi->currentStep << "/" << psi->maxSteps
                      		<< " simulating time: " << g_simulationStep * deltaT << endl;
                  	count = 0;
              	}

              	count++; )

#ifdef PERFORMANCE_METRICS
		cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
		normalMTGPU(randNoise_d);
#ifdef PERFORMANCE_METRICS
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		t_gpu_rndGeneration += time;
#endif // PERFORMANCE_METRICS

		// display running info to console
		// Advance neurons ------------->
		blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
#ifdef PERFORMANCE_METRICS
		cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
#ifdef STORE_SPIKEHISTORY
		advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, spikeHistory_d, g_simulationStep, maxSpikes, delayIdx.getIndex(), maxSynapses );
#else
		advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, g_simulationStep, delayIdx.getIndex(), maxSynapses );
#endif // STORE_SPIKEHISTORY
#ifdef PERFORMANCE_METRICS
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		t_gpu_advanceNeurons += time;
#endif // PERFORMANCE_METRICS

		// Advance synapses ------------->
		blocksPerGrid = ( synapse_count + threadsPerBlock - 1 ) / threadsPerBlock;
#ifdef PERFORMANCE_METRICS
		cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
		advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( synapse_count, width, g_simulationStep, delayIdx.getBitmask() );
#ifdef PERFORMANCE_METRICS
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		t_gpu_advanceSynapses += time;
#endif // PERFORMANCE_METRICS

		// calculate summation point
		blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
#ifdef PERFORMANCE_METRICS
		cudaEventRecord( start, 0 );
#endif // PERFORMANCE_METRICS
		calcSummationMap <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, inverseMap_d );
#ifdef PERFORMANCE_METRICS
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop );
		t_gpu_calcSummation += time;
#endif // PERFORMANCE_METRICS

		// Advance the clock
		g_simulationStep++;
		// Advance the delayed queue index
		delayIdx.inc();
	}
#ifdef PERFORMANCE_METRICS
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	cout << endl;
	cout << "neuron_count: " << neuron_count << endl;
	cout << "synapse_count: " << synapse_count << endl;
	cout << "  effective bandwidth: " << 
		getEffectiveBandwidth( count, neuron_count * 44 + synapse_count * 20, neuron_count * 8, t_gpu_calcSummation ) << " GB/s" << endl;
#endif // PERFORMANCE_METRICS
	cout << endl;

#ifdef STORE_SPIKEHISTORY
	// Copy processed data from GPU device memory to host
	cudaDeviceSynchronize();
	HANDLE_ERROR( cudaMemcpy ( spikeArray, spikeHistory_d, spikeHistory_d_size, cudaMemcpyDeviceToHost ) );
#endif // STORE_SPIKEHISTORY

	DEBUG(cout << "Completed GPU sim cycle" << endl;)
}


/**
 * @param[in] psi		Pointer to the simulation information.
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
void createSynapseImap(SimulationInfo * psi, uint32_t maxSynapses )
{
	LifNeuron_struct neuron_st;
	LifSynapse_struct synapse_st;
	uint32_t neuron_count = psi->cNeurons;
	uint32_t synapse_count = 0;

	// copy device neuron struct to host memory
	allocNeuronStruct( neuron_st, neuron_count );
	copyNeuronDeviceToHost( neuron_st, neuron_count );

	// count the synapses
	for ( uint32_t i = 0; i < neuron_count; i++ )
	{
		assert( neuron_st.synapseCount[i] < maxSynapses );
		synapse_count += neuron_st.synapseCount[i];
	}

	DEBUG ( cout << "synapse_count: " << synapse_count << endl; )

	if ( synapse_count == 0 )
	{
		deleteNeuronStruct(neuron_st);
		return;
	}

	// copy device synapse struct to host memory 
	// PAB: (I think below line is unnecessary because memory has already been allocated for synapses)
	allocSynapseSumCoord( synapse_st, maxSynapses * neuron_count );
	copySynapseSumCoordDeviceToHost( synapse_st, maxSynapses * neuron_count );

	// allocate memories for inverse map
	vector<uint32_t>* rgSynapseInverseMap = new vector<uint32_t>[neuron_count];
	uint32_t* rgIverseMap = new uint32_t[synapse_count];

	uint32_t syn_i = 0;
	uint32_t n_inUse = 0;

	for (uint32_t i = 0; i < neuron_count; i++)
	{
		for ( uint32_t j = 0; j < maxSynapses; j++, syn_i++ )
		{
			if ( synapse_st.inUse[syn_i] == true )
			{
				uint32_t idx = synapse_st.summationCoord[syn_i].x 
					+ synapse_st.summationCoord[syn_i].y * psi->width;
				rgSynapseInverseMap[idx].push_back(syn_i);
				DEBUG ( n_inUse++; )
			}
		}
	}

	DEBUG ( assert( synapse_count == n_inUse ); )

	// create synapse inverse map
	syn_i = 0;
	for (uint32_t i = 0; i < neuron_count; i++)
	{
		neuron_st.incomingSynapse_begin[i] = syn_i;
		neuron_st.inverseCount[i] = rgSynapseInverseMap[i].size();

		for ( uint32_t j = 0; j < rgSynapseInverseMap[i].size(); j++, syn_i++)
		{
			rgIverseMap[syn_i] = rgSynapseInverseMap[i][j];
		}
	}

	// copy inverse map to the device memory
	if ( inverseMap_d != NULL)
		HANDLE_ERROR( cudaFree( inverseMap_d ) );
	size_t inverseMap_d_size = synapse_count * sizeof(uint32_t);
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &inverseMap_d, inverseMap_d_size ) );
	HANDLE_ERROR( cudaMemcpy ( inverseMap_d, rgIverseMap, inverseMap_d_size, cudaMemcpyHostToDevice ) );

	// update imap information in neuron struct
	LifNeuron_struct neuron;
	HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.incomingSynapse_begin, neuron_st.incomingSynapse_begin, neuron_count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.inverseCount, neuron_st.inverseCount, neuron_count * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );

	// delete memories
	deleteNeuronStruct( neuron_st );
	deleteSynapseSumCoord( synapse_st );
	delete[] rgSynapseInverseMap;
	delete[] rgIverseMap;
}

#ifdef PERFORMANCE_METRICS
/**
 * Calculate effective bandwidth in GB/s
 *
 * @param count	number of calls of the kernel
 * @param Br	number of bytes read per kernel
 * @param Bw	number of bytes written per kernel
 * @param time	total elapsed time in ms
 */
float getEffectiveBandwidth( uint64_t count, uint32_t Br, uint32_t Bw, float time ) {
	return ( ( Br + Bw ) * count / 1e6 ) / time;
}
#endif // PERFORMANCE_METRICS


// CUDA code for advancing neurons-----------------------------------------------------------------------
#ifdef STORE_SPIKEHISTORY
/**
 * @param[in] n			Number of synapses.
 * @param[in] spikeHistory_d	Spike history list.
 * @param[in] simulationStep	The current simulation step.
 * @param[in] maxSpikes		Maximum number of spikes per neuron per one epoch.
 * @param[in] delay   		Index of the delayed list (spike queue).
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
__global__ void advanceNeuronsDevice( uint32_t n, uint64_t* spikeHistory_d, uint64_t simulationStep, uint32_t maxSpikes, uint32_t delay, uint32_t maxSynapses )
#else
/**
 * @param[in] n		    	Number of synapses.
 * @param[in] simulationStep	The current simulation step.
 * @param[in] delay  		Index of the delayed list (spike queue).
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
__global__ void advanceNeuronsDevice( uint32_t n, uint64_t simulationStep, uint32_t delay, uint32_t maxSynapses )
#endif // STORE_SPIKEHISTORY
{
	// determine which neuron this thread is processing
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
		return;

	neuron_st_d[0].hasFired[idx] = false;
	BGFLOAT& sp = *neuron_st_d[0].summationPoint[idx];
	BGFLOAT& vm = neuron_st_d[0].Vm[idx];
	BGFLOAT r_sp = sp;
	BGFLOAT r_vm = vm;

	if ( neuron_st_d[0].nStepsInRefr[idx] > 0 ) { // is neuron refractory?
		--neuron_st_d[0].nStepsInRefr[idx];
	} else if ( r_vm >= neuron_st_d[0].Vthresh[idx] ) { // should it fire?
		// Note that the neuron has fired!
		neuron_st_d[0].hasFired[idx] = true;

#ifdef STORE_SPIKEHISTORY
		// record spike time
		spikeHistory_d[(idx * maxSpikes) + neuron_st_d[0].spikeCount[idx]] = simulationStep;
#endif // STORE_SPIKEHISTORY
		neuron_st_d[0].spikeCount[idx]++;

		// calculate the number of steps in the absolute refractory period
		neuron_st_d[0].nStepsInRefr[idx] = static_cast<uint32_t> ( neuron_st_d[0].Trefract[idx] / neuron_st_d[0].deltaT[idx] + 0.5 );

		// reset to 'Vreset'
		vm = neuron_st_d[0].Vreset[idx];

		// notify synapses of spike
		uint32_t syn_i = neuron_st_d[0].outgoingSynapse_begin[idx];
		for ( uint32_t i = 0; i < maxSynapses; i++ ) {
			if ( synapse_st_d[0].inUse[syn_i + i] == true )
			{
				// notify synapses of spike...
				uint32_t idx0 = delay + synapse_st_d[0].total_delay[syn_i + i];
				if ( idx0 >= LENGTH_OF_DELAYQUEUE )
					idx0 -= LENGTH_OF_DELAYQUEUE;

				// set a spike
				synapse_st_d[0].delayQueue[syn_i + i] |= (0x1 << idx0);
			}
		}
	} else {

		r_sp += neuron_st_d[0].I0[idx]; // add IO
		
		// Random number alg. goes here    
		r_sp += (*neuron_st_d[0].randNoise[idx] * neuron_st_d[0].Inoise[idx]); // add cheap noise
		vm = neuron_st_d[0].C1[idx] * r_vm + neuron_st_d[0].C2[idx] * ( r_sp ); // decay Vm and add inputs
	}

	// clear synaptic input for next time step
	sp = 0;
}

// CUDA code for advancing synapses ---------------------------------------------------------------------
/**
 * @param[in] n			Number of synapses.
 * @param[in] width		Width of neuron map (assumes square).
 * @param[in] simulationStep	The current simulation step.
 */
__global__ void advanceSynapsesDevice ( uint32_t n, uint32_t width, uint64_t simulationStep, uint32_t bmask ) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
		return;

	if ( synapse_st_d[0].inUse[idx] != true )
		return;

	uint32_t itype = synapse_st_d[0].type[idx];

	// is there a spike in the queue?
	uint32_t s_delayQueue = synapse_st_d[0].delayQueue[idx];
	bool isFired = s_delayQueue & bmask;
	synapse_st_d[0].delayQueue[idx] = s_delayQueue & (~bmask);
	BGFLOAT s_decay = synapse_st_d[0].decay[idx];
	if ( isFired ) {
		// adjust synapse paramaters
		if ( synapse_st_d[0].lastSpike[idx] != ULONG_MAX ) {
			BGFLOAT isi = (simulationStep - synapse_st_d[0].lastSpike[idx]) * synapse_st_d[0].deltaT[idx];
			synapse_st_d[0].r[idx] = 1 + ( synapse_st_d[0].r[idx] * ( 1 - synapse_st_d[0].u[idx] ) - 1 ) * exp ( -isi / synapse_D_d[itype] );
			synapse_st_d[0].u[idx] = synapse_U_d[itype] + synapse_st_d[0].u[idx] * ( 1 - synapse_U_d[itype] ) * exp ( -isi / synapse_F_d[itype] );
		}

		synapse_st_d[0].psr[idx] += ( ( synapse_st_d[0].W[idx] / s_decay ) * synapse_st_d[0].u[idx] * synapse_st_d[0].r[idx] );// calculate psr
		synapse_st_d[0].lastSpike[idx] = simulationStep; // record the time of the spike
	}

	// decay the post spike response
	synapse_st_d[0].psr[idx] *= s_decay;
}

// CUDA code for calculating summation point -----------------------------------------------------
/**
 * @param[in] n			Number of neurons.
 * @param[in] inverseMap	Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron..
 */
__global__ void calcSummationMap( uint32_t n, uint32_t* inverseMap ) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= n || inverseMap == NULL)
                return;
        
        uint32_t* inverseMap_begin = &inverseMap[neuron_st_d[0].incomingSynapse_begin[idx]];
        BGFLOAT sum = 0.0;
        uint32_t iCount = neuron_st_d[0].inverseCount[idx];
        for ( uint32_t i = 0; i < iCount; i++ ) {
                uint32_t syn_i = inverseMap_begin[i];
                sum += synapse_st_d[0].psr[syn_i];
        }
        *neuron_st_d[0].summationPoint[idx] = sum;
} 

// CUDA code for calculating neuron/synapse offsets -----------------------------------------------------
/**
 * @param[in] n			Number of neurons.
 * @param[in] summationPoint_d	The summation map.
 * @param[in] width		Width of neuron map (assumes square).
 * @param[in] randNoise_d	Array of randum numbers. 
 */
__global__ void calcOffsets( uint32_t n, BGFLOAT* summationPoint_d, uint32_t width, float* randNoise_d )
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
		return;

	// set summation pointer
	neuron_st_d[0].summationPoint[idx] = &summationPoint_d[idx];
	*neuron_st_d[0].summationPoint[idx] = 0;

	neuron_st_d[0].randNoise[idx] = &randNoise_d[idx];
}

/**
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below 
 * zero.
 * @param[in] summationPoint_d	The summation map.
 * @param[in] rgNeuronTypeMap_d	The neuron type map (INH, EXC).
 * @param[in] n			Number of neurons.
 * @param[in] width		Width of neuron map (assumes square).
 * @param[in] deltaT		The time step size.
 * @param[in] W_d		Array of synapse weight.
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
__global__ void updateNetworkDevice( BGFLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, uint32_t n, uint32_t width, BGFLOAT deltaT, BGFLOAT* W_d, uint32_t maxSynapses )
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
		return;

	uint32_t adjusted = 0;
	uint32_t removed = 0;
	uint32_t added = 0;

	// source neuron coordinate
	uint32_t a = idx;
	uint32_t xa = a % width;
	uint32_t ya = a / width;

	// and each destination neuron 'b'
	for ( uint32_t b = 0; b < n; b++ )
	{
		uint32_t xb = b % width;
		uint32_t yb = b / width;

		// visit each synapse at (xa, ya)
		bool connected = false;

		// for each existing synapse
		uint32_t syn_i = neuron_st_d[0].outgoingSynapse_begin[a];
		for ( uint32_t i = 0; i < maxSynapses; i++ )
		{
			if ( synapse_st_d[0].inUse[syn_i + i] != true)
				continue;
			// if there is a synapse between a and b
			if ( synapse_st_d[0].summationCoord[syn_i + i].x == xb &&
				synapse_st_d[0].summationCoord[syn_i + i].y == yb )
			{
				connected = true;
				adjusted++;

				// adjust the strength of the synapse or remove 
				// it from the synapse map if it has gone below
				// zero.
				if ( W_d[a * n + b] < 0 )
				{
					removed++;
					removeSynapse( a, syn_i + i );
				}
				else
				{
					// adjust
					// g_synapseStrengthAdjustmentConstant is 1.0e-8;
					synapse_st_d[0].W[syn_i + i] = W_d[a * n + b] 
						* synSign( synType( rgNeuronTypeMap_d, xa, ya, xb, yb, width ) ) 
						* g_synapseStrengthAdjustmentConstant_d;
				}
			}
		}

		// if not connected and weight(a,b) > 0, add a new synapse from a to b
		if ( !connected && ( W_d[a * n + b] > 0 ) )
		{
			added++;
			BGFLOAT W_new = W_d[a * n + b] 
				* synSign( synType( rgNeuronTypeMap_d, xa, ya, xb, yb, width ) ) 
				* g_synapseStrengthAdjustmentConstant_d;	
			addSynapse( W_new, summationPoint_d, rgNeuronTypeMap_d, a, xa, ya, xb, yb, width, deltaT, maxSynapses );
		}
	}
}

/**
* Adds a synapse to the network.  Requires the locations of the source and
* destination neurons.
* @param W_new			The weight (scaling factor, strength, maximal amplitude) of the synapse.
* @param summationPoint_d	The summagtion map.
* @param rgNeuronTypeMap_d	The neuron type map (INH, EXC).
* @param neuron_i		Index of the source neuron.
* @param source_x		X location of source.
* @param source_y		Y location of source.
* @param dest_x			X location of destination.
* @param dest_y			Y location of destination.
* @param width			Width of neuron map (assumes square).
* @param deltaT			The time step size.
* @param maxSynapses		Maximum number of synapses per neuron.
*/
__device__ void addSynapse( BGFLOAT W_new, BGFLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, uint32_t neuron_i, uint32_t source_x, uint32_t source_y, uint32_t dest_x, uint32_t dest_y, uint32_t width, BGFLOAT deltaT, uint32_t maxSynapses )
{
	if ( neuron_st_d[0].synapseCount[neuron_i] >= maxSynapses )
		return;			// TODO: ERROR!

	// locate summation point
	BGFLOAT* sp = &( summationPoint_d[dest_x + dest_y * width] );

	// determine the synapse type
	synapseType type = synType( rgNeuronTypeMap_d, source_x, source_y, dest_x, dest_y, width );

	// add it to the list
	uint32_t syn_i = neuron_st_d[0].outgoingSynapse_begin[neuron_i];
	for ( uint32_t i = 0; i < maxSynapses; i++, syn_i++ )
		if ( synapse_st_d[0].inUse[syn_i] != true )
			break;

	neuron_st_d[0].synapseCount[neuron_i]++;

	// create a synapse
	createSynapse( syn_i, source_x, source_y, dest_x, dest_y, sp, deltaT, type );	
	synapse_st_d[0].W[syn_i] = W_new;
}

/**
* Create a synapse.
* @param syn_i		Index of the synapse.
* @param source_x	X location of source.
* @param source_y	Y location of source.
* @param dest_x		X location of destination.
* @param dest_y		Y location of destination.
* @param sp		Summation point.
* @param deltaT		The time step size.
* @param type		Type of the synapse.
*/
__device__ void createSynapse( uint32_t syn_i, uint32_t source_x, uint32_t source_y, uint32_t dest_x, uint32_t dest_y, BGFLOAT* sp, BGFLOAT deltaT, synapseType type )
{
	BGFLOAT delay;

	synapse_st_d[0].inUse[syn_i] = true;
	synapse_st_d[0].summationPoint[syn_i] = sp;
	synapse_st_d[0].summationCoord[syn_i].x = dest_x;
	synapse_st_d[0].summationCoord[syn_i].y = dest_y;
	synapse_st_d[0].synapseCoord[syn_i].x = source_x;	
	synapse_st_d[0].synapseCoord[syn_i].y = source_y;	
	synapse_st_d[0].deltaT[syn_i] = deltaT;
	synapse_st_d[0].W[syn_i] = 10.0e-9;
	synapse_st_d[0].psr[syn_i] = 0.0;
	synapse_st_d[0].delayQueue[syn_i] = 0;
	synapse_st_d[0].r[syn_i] = 1.0;
	synapse_st_d[0].u[syn_i] = 0.4;		// DEFAULT_U
	synapse_st_d[0].lastSpike[syn_i] = ULONG_MAX;
	synapse_st_d[0].type[syn_i] = type;

	BGFLOAT tau;
	switch ( type ) {
	case II:
		tau = 6e-3;
		delay = 0.8e-3;
		break;
	case IE:
		tau = 6e-3;
		delay = 0.8e-3;
		break;
	case EI:
		tau = 3e-3;
		delay = 0.8e-3;
		break;
	case EE:
		tau = 3e-3;
		delay = 1.5e-3;
		break;
	}

	synapse_st_d[0].tau[syn_i] = tau;
	synapse_st_d[0].total_delay[syn_i] = static_cast<uint32_t>( delay / deltaT ) + 1;
	synapse_st_d[0].decay[syn_i] = exp( -deltaT / tau );
	//PAB: This diverges from Single-Threaded behavior -- an initSpikeQueue would be performed here
}

/**
* Remove a synapse from the network.
* @param neuron_i	Index of a neuron.
* @param syn_i		Index of a synapse.
*/
__device__ void removeSynapse( uint32_t neuron_i, uint32_t syn_i )
{
	neuron_st_d[0].synapseCount[neuron_i]--;
	synapse_st_d[0].inUse[syn_i] = false;
}

/**
* Returns the type of synapse at the given coordinates
* @param rgNeuronTypeMap_d	The neuron type map (INH, EXC).
* @param ax	Source coordinate(x).
* @param ay	Source coordinate(y).
* @param bx	Destination coordinate(x).
* @param by	Destination coordinate(y).
* @param width	Width of neuron map (assumes square).
* @return type of synapse at the given coordinate or -1 on error
*/
__device__ synapseType synType( neuronType* rgNeuronTypeMap_d, uint32_t ax, uint32_t ay, uint32_t bx, uint32_t by, uint32_t width )
{
	if ( rgNeuronTypeMap_d[ax + ay * width] == INH && rgNeuronTypeMap_d[bx + by * width] == INH )
		return II;
	else if ( rgNeuronTypeMap_d[ax + ay * width] == INH && rgNeuronTypeMap_d[bx + by * width] == EXC )
		return IE;
	else if ( rgNeuronTypeMap_d[ax + ay * width] == EXC && rgNeuronTypeMap_d[bx + by * width] == INH )
		return EI;
	else if ( rgNeuronTypeMap_d[ax + ay * width] == EXC && rgNeuronTypeMap_d[bx + by * width] == EXC )
		return EE;

	return STYPE_UNDEF;
}

/**
* Return 1 if originating neuron is excitatory, -1 otherwise.
* @param[in] t	synapseType I to I, I to E, E to I, or E to E
* @return 1 or -1
*/
__device__ int32_t synSign( synapseType t )
{
	switch ( t )
	{
	case II:
	case IE:
		return -1;
	case EI:
	case EE:
		return 1;
	}

	return 0;
}

