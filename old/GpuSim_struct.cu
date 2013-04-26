/**
 ** \file GpuSim_struct.cu
 **
 ** \authors Fumitaka Kawasaki
 **
 ** \brief Functions that perform the GPU version of simulation.
 **/

#include "global.h"
#include "SimulationInfo.h"
#include "DynamicSpikingSynapse_struct.h"
#include "LifNeuron_struct.h"
#include "book.h"
#include "DelayIdx.h"
#include "matrix/VectorMatrix.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Forward Declarations
extern "C" {
//! Perform updating neurons and synapses for one activity epoch.
void advanceGPU( 
#ifdef STORE_SPIKEHISTORY
		SimulationInfo* psi,
		int maxSynapses, 
		uint64_t* spikeArray,
		int maxSpikes
#else
		SimulationInfo* psi,
		int maxSynapses 
#endif // STORE_SPIKEHISTORY
		 );

//! Allocate GPU device memory and copy data from host memory.
void allocDeviceStruct( SimulationInfo* psi, 
		LifNeuron_struct& neuron_st, 
		DynamicSpikingSynapse_struct& synapse_st,
#ifdef STORE_SPIKEHISTORY
                int maxSynapses,
		int maxSpikes 
#else
                int maxSynapses
#endif // STORE_SPIKEHISTORY
		);

void copySynapseDeviceToHost( DynamicSpikingSynapse_struct& synapse_h, int count );

void copyNeuronDeviceToHost( LifNeuron_struct& neuron_h, int count );

//! Deallocate device memory.
void deleteDeviceStruct( );

//! Get spike count of each neuron.
void getSpikeCounts( int neuron_count, int* spikeCounts );

//! Clear spike count of each neuron.
void clearSpikeCounts( int neuron_count );

//! Update the network.
void updateNetworkGPU( SimulationInfo* psi, CompleteMatrix& W, int maxSynapses );

//! Create synapse inverse map.
void createSynapseImap( SimulationInfo* psi, int maxSynapses );

//! Generate random number (normal distribution)
void normalMTGPU(float * randNoise_d);
}

#ifdef STORE_SPIKEHISTORY
//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( int n, uint64_t* spikeHistory_d, uint64_t simulationStep, int maxSpikes, int delayIdx, int maxSynapses );
#else
//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( int n, uint64_t simulationStep, int delayIdx, int maxSynapses );
#endif // STORE_SPIKEHISTORY
//! Perform updating synapses for one time step.
__global__ void advanceSynapsesDevice( int n, int width, uint64_t simulationStep, uint32_t bmask );

//! Calculate neuron/synapse offsets.
__global__ void calcOffsets( int n, FLOAT* summationPoint_d, int width, float* randNoise_d );

//! Calculate summation point.
__global__ void calcSummationMap( int n, uint32_t* inverseMap );

//! Update the network.
__global__ void updateNetworkDevice( FLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, int n, int width, FLOAT deltaT, FLOAT* W_d, int maxSynapses );

//! Add a synapse to the network.
__device__ void addSynapse( FLOAT W_new, FLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, int neuron_i, int source_x, int source_y, int dest_x, int dest_y, int width, FLOAT deltaT, int maxSynapses );

//! Create a synapse.
__device__ void createSynapse( int syn_i, int source_x, int source_y, int dest_x, int dest_y, FLOAT* sp, FLOAT deltaT, synapseType type );

//! Remove a synapse from the network.
__device__ void removeSynapse( int neuron_i, int syn_i );

//! Get the type of synapse.
__device__ synapseType synType( neuronType* rgNeuronTypeMap_d, int ax, int ay, int bx, int by, int width );

//! Get the type of synapse (excitatory or inhibitory)
__device__ int synSign( synapseType t );

#ifdef PERFORMANCE_METRICS
//! Calculate effective bandwidth in GB/s.
float getEffectiveBandwidth( uint64_t count, int Br, int Bw, float time );
#endif // PERFORMANCE_METRICS

//! Delayed queue index - global to all synapses.
DelayIdx delayIdx;

//! Synapse constant (U)stored in device constant memory.
__constant__ FLOAT synapse_U_d[4] = { 0.32, 0.25, 0.05, 0.5 };	// II, IE, EI, EE

//! Synapse constant(D) stored in device constant memory.
__constant__ FLOAT synapse_D_d[4] = { 0.144, 0.7, 0.125, 1.1 };	// II, IE, EI, EE

//! Synapse constant(F) stored in device constant memory.
__constant__ FLOAT synapse_F_d[4] = { 0.06, 0.02, 1.2, 0.05 };	// II, IE, EI, EE

//! Neuron structure in device constant memory.
__constant__ LifNeuron_struct neuron_st_d[1];

//! Synapse structures in device constant memory.
__constant__ DynamicSpikingSynapse_struct synapse_st_d[1];

__constant__ FLOAT g_synapseStrengthAdjustmentConstant_d = 1.0e-8;

#include "LifNeuron_struct_d.cu"
#include "DynamicSpikingSynapse_struct_d.cu"

#ifdef STORE_SPIKEHISTORY
//! Pointer to device spike history array.
uint64_t* spikeHistory_d = NULL;	
#endif // STORE_SPIKEHISTORY

//! Pointer to device summation point.
FLOAT* summationPoint_d = NULL;	

//! Pointer to device random noise array.
float* randNoise_d = NULL;	

//! Pointer to device inverse map.
uint32_t* inverseMap_d = NULL;	

//! Pointer to neuron type map.
neuronType* rgNeuronTypeMap_d = NULL;

/**
 * @param[in] psi		Pointer to the simulation information.
 * @param[in] neuron_st		A leaky-integrate-and-fire (I&F) neuron structure.
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 * @param[in] maxSpikes		Maximum number of spikes per neuron per one epoch.
 */
void allocDeviceStruct( SimulationInfo* psi, 
		LifNeuron_struct& neuron_st,
		DynamicSpikingSynapse_struct& synapse_st,
#ifdef STORE_SPIKEHISTORY
                int maxSynapses,
		int maxSpikes 
#else
                int maxSynapses
#endif // STORE_SPIKEHISTORY
		)
{
	// Set device ID
	HANDLE_ERROR( cudaSetDevice( g_deviceId ) );

	// CUDA parameters
	const int threadsPerBlock = 256;
	int blocksPerGrid;

	// Allocate GPU device memory
	int neuron_count = psi->cNeurons;
	int synapse_count = neuron_count * maxSynapses;
	allocNeuronStruct_d( neuron_count );				// and allocate device memory for each member
	allocSynapseStruct_d( synapse_count );				// and allocate device memory for each member

#ifdef STORE_SPIKEHISTORY
	size_t spikeHistory_d_size = neuron_count * maxSpikes * sizeof (uint64_t);		// size of spike history array
#endif // STORE_SPIKEHISTORY
	size_t summationPoint_d_size = neuron_count * sizeof (FLOAT);	// size of summation point
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
	HANDLE_ERROR( cudaMemcpy ( rgNeuronTypeMap_d, psi->rgNeuronTypeMap, rgNeuronTypeMap_d_size, cudaMemcpyHostToDevice ) );

	int width = psi->width;	
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
void advanceGPU( SimulationInfo* psi, int maxSynapses, uint64_t* spikeArray, int maxSpikes )
#else
/**
 * @param[in] psi		Pointer to the simulation information. 
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
void advanceGPU( SimulationInfo* psi, int maxSynapses )
#endif
{
	FLOAT deltaT = psi->deltaT;
	int width = psi->width;
	int neuron_count = psi->cNeurons;
	int synapse_count = neuron_count * maxSynapses;

        // simulate to next growth cycle
        uint64_t endStep = g_simulationStep + static_cast<uint64_t>(psi->stepDuration / deltaT);
	
	DEBUG(cout << "Beginning GPU sim cycle, simTime = " << g_simulationStep * deltaT << ", endTime = " << endStep * deltaT << endl;)

	// CUDA parameters
	const int threadsPerBlock = 256;
	int blocksPerGrid;

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
        	DEBUG( if (count %1000 == 0)
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
		uint32_t bmask = delayIdx.getBitmask(  );
		advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( synapse_count, width, g_simulationStep, bmask );
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
	size_t spikeHistory_d_size = neuron_count * maxSpikes * sizeof (uint64_t);		// size of spike history array
	HANDLE_ERROR( cudaMemcpy ( spikeArray, spikeHistory_d, spikeHistory_d_size, cudaMemcpyDeviceToHost ) );
#endif // STORE_SPIKEHISTORY

	DEBUG(cout << "Completed GPU sim cycle" << endl;)
}

/**
 * @param[in] neuron_count	Number of neurons.
 * @param[out] spikeCounts	Array to store spike counts for neurons. 
 */
void getSpikeCounts( int neuron_count, int* spikeCounts )
{
	LifNeuron_struct neuron;
        HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );
	HANDLE_ERROR( cudaMemcpy ( spikeCounts, neuron.spikeCount, neuron_count * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/**
 * @param[in] neuron_count	Number of neurons.
 */
void clearSpikeCounts( int neuron_count )
{
	LifNeuron_struct neuron;
        HANDLE_ERROR( cudaMemcpyFromSymbol ( &neuron, neuron_st_d, sizeof( LifNeuron_struct ) ) );
	HANDLE_ERROR( cudaMemset( neuron.spikeCount, 0, neuron_count * sizeof( int ) ) );
}

/**
 * @param[in] psi		Pointer to the simulation information.
 * @param[in] W			Array of synapse weight.
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
void updateNetworkGPU( SimulationInfo* psi, CompleteMatrix& W, int maxSynapses )
{
	int neuron_count = psi->cNeurons;
	int width = psi->width;
	FLOAT deltaT = psi->deltaT;

        // CUDA parameters
        const int threadsPerBlock = 256;
        int blocksPerGrid;

	// allocate memories
	size_t W_d_size = neuron_count * neuron_count * sizeof (FLOAT);
	FLOAT* W_h = new FLOAT[W_d_size];
	FLOAT* W_d;
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

	// copy weight data to the device memory
	for ( int i = 0 ; i < neuron_count; i++ )
		for ( int j = 0; j < neuron_count; j++ )
			W_h[i * neuron_count + j] = W(i, j);

	HANDLE_ERROR( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

	blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
	updateNetworkDevice <<< blocksPerGrid, threadsPerBlock >>> ( summationPoint_d, rgNeuronTypeMap_d, neuron_count, width, deltaT, W_d, maxSynapses );

	// free memories
	HANDLE_ERROR( cudaFree( W_d ) );
	delete[] W_h;

	// create synapse inverse map
	createSynapseImap( psi, maxSynapses );
}

/**
 * @param[in] psi		Pointer to the simulation information.
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
void createSynapseImap( SimulationInfo* psi, int maxSynapses )
{
	LifNeuron_struct neuron_st;
	DynamicSpikingSynapse_struct synapse_st;
	int neuron_count = psi->cNeurons;
	int synapse_count = 0;

	// copy device neuron struct to host memory
	allocNeuronStruct( neuron_st, neuron_count );
	copyNeuronDeviceToHost( neuron_st, neuron_count );

	// count the synapses
	for ( int i = 0; i < neuron_count; i++ )
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
	allocSynapseSumCoord( synapse_st, maxSynapses * neuron_count );
	copySynapseSumCoordDeviceToHost( synapse_st, maxSynapses * neuron_count );

	// allocate memories for inverse map
	vector<uint32_t>* rgSynapseInverseMap = new vector<uint32_t>[neuron_count];
	uint32_t* rgIverseMap = new uint32_t[synapse_count];

	uint32_t syn_i = 0;
	DEBUG ( int n_inUse = 0; )

	for (int i = 0; i < neuron_count; i++)
	{
		for ( int j = 0; j < maxSynapses; j++, syn_i++ )
		{
			if ( synapse_st.inUse[syn_i] == true )
			{
				int idx = synapse_st.summationCoord[syn_i].x 
					+ synapse_st.summationCoord[syn_i].y * psi->width;
				rgSynapseInverseMap[idx].push_back(syn_i);
				DEBUG ( n_inUse++; )
			}
		}
	}

	DEBUG ( assert( synapse_count == n_inUse ); )

	// create synapse inverse map
	syn_i = 0;
	for (int i = 0; i < neuron_count; i++)
	{
		neuron_st.incomingSynapse_begin[i] = syn_i;
		neuron_st.inverseCount[i] = rgSynapseInverseMap[i].size();

		for ( int j = 0; j < rgSynapseInverseMap[i].size(); j++, syn_i++)
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
	HANDLE_ERROR( cudaMemcpy ( neuron.incomingSynapse_begin, neuron_st.incomingSynapse_begin, neuron_count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( neuron.inverseCount, neuron_st.inverseCount, neuron_count * sizeof( int ), cudaMemcpyHostToDevice ) );

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
float getEffectiveBandwidth( uint64_t count, int Br, int Bw, float time ) {
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
 * @param[in] delayIdx		Index of the delayed list (spike queue).
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
__global__ void advanceNeuronsDevice( int n, uint64_t* spikeHistory_d, uint64_t simulationStep, int maxSpikes, int delayIdx, int maxSynapses ) {
#else
/**
 * @param[in] n			Number of synapses.
 * @param[in] simulationStep	The current simulation step.
 * @param[in] delayIdx		Index of the delayed list (spike queue).
 * @param[in] maxSynapses	Maximum number of synapses per neuron.
 */
__global__ void advanceNeuronsDevice( int n, uint64_t simulationStep, int delayIdx, int maxSynapses ) {
#endif // STORE_SPIKEHISTORY
	// determine which neuron this thread is processing
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
		return;

	neuron_st_d[0].hasFired[idx] = false;
	FLOAT& sp = *neuron_st_d[0].summationPoint[idx];
	FLOAT& vm = neuron_st_d[0].Vm[idx];
	FLOAT r_sp = sp;
	FLOAT r_vm = vm;

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
		neuron_st_d[0].nStepsInRefr[idx] = static_cast<int> ( neuron_st_d[0].Trefract[idx] / neuron_st_d[0].deltaT[idx] + 0.5 );

		// reset to 'Vreset'
		vm = neuron_st_d[0].Vreset[idx];

		// notify synapses of spike
		int syn_i = neuron_st_d[0].outgoingSynapse_begin[idx];
		for ( int i = 0; i < maxSynapses; i++ ) {
			if ( synapse_st_d[0].inUse[syn_i + i] == true )
			{
				// notify synapses of spike...
				int idx0 = delayIdx + synapse_st_d[0].total_delay[syn_i + i];
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
 * @param[in] bmask		Bit mask for the delayed list (spike queue).
 */
__global__ void advanceSynapsesDevice ( int n, int width, uint64_t simulationStep, uint32_t bmask ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
		return;

	if ( synapse_st_d[0].inUse[idx] != true )
		return;

	int itype = synapse_st_d[0].type[idx];

	// is there a spike in the queue?
	uint32_t s_delayQueue = synapse_st_d[0].delayQueue[idx];
	bool isFired = s_delayQueue & bmask;
	synapse_st_d[0].delayQueue[idx] = s_delayQueue & (~bmask);
	FLOAT s_decay = synapse_st_d[0].decay[idx];
	if ( isFired ) {
		// adjust synapse paramaters
		if ( synapse_st_d[0].lastSpike[idx] != ULONG_MAX ) {
			FLOAT isi = (simulationStep - synapse_st_d[0].lastSpike[idx]) * synapse_st_d[0].deltaT[idx];
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
__global__ void calcSummationMap( int n, uint32_t* inverseMap ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= n )
                return;
        
        uint32_t* inverseMap_begin = &inverseMap[neuron_st_d[0].incomingSynapse_begin[idx]];
        FLOAT sum = 0.0;
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
__global__ void calcOffsets( int n, FLOAT* summationPoint_d, int width, float* randNoise_d )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void updateNetworkDevice( FLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, int n, int width, FLOAT deltaT, FLOAT* W_d, int maxSynapses )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= n )
		return;

	int adjusted = 0;
	int removed = 0;
	int added = 0;

	// source neuron coordinate
	int a = idx;
	int xa = a % width;
	int ya = a / width;

	// and each destination neuron 'b'
	for ( int b = 0; b < n; b++ )
	{
		int xb = b % width;
		int yb = b / width;

		// visit each synapse at (xa, ya)
		bool connected = false;

		// for each existing synapse
		int syn_i = neuron_st_d[0].outgoingSynapse_begin[a];
		for ( int i = 0; i < maxSynapses; i++ )
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
			FLOAT W_new = W_d[a * n + b] 
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
__device__ void addSynapse( FLOAT W_new, FLOAT* summationPoint_d, neuronType* rgNeuronTypeMap_d, int neuron_i, int source_x, int source_y, int dest_x, int dest_y, int width, FLOAT deltaT, int maxSynapses )
{
	if ( neuron_st_d[0].synapseCount[neuron_i] >= maxSynapses )
		return;			// TODO: ERROR!

	// locate summation point
	FLOAT* sp = &( summationPoint_d[dest_x + dest_y * width] );

	// determine the synapse type
	synapseType type = synType( rgNeuronTypeMap_d, source_x, source_y, dest_x, dest_y, width );

	// add it to the list
	int syn_i = neuron_st_d[0].outgoingSynapse_begin[neuron_i];
	for ( int i = 0; i < maxSynapses; i++, syn_i++ )
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
__device__ void createSynapse( int syn_i, int source_x, int source_y, int dest_x, int dest_y, FLOAT* sp, FLOAT deltaT, synapseType type )
{
	FLOAT delay;

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
	synapse_st_d[0].ldelayQueue[syn_i] = LENGTH_OF_DELAYQUEUE;
	synapse_st_d[0].r[syn_i] = 1.0;
	synapse_st_d[0].u[syn_i] = 0.4;		// DEFAULT_U
	synapse_st_d[0].lastSpike[syn_i] = ULONG_MAX;
	synapse_st_d[0].type[syn_i] = type;

	FLOAT tau;
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
	synapse_st_d[0].total_delay[syn_i] = static_cast<int>( delay / deltaT ) + 1;
	synapse_st_d[0].decay[syn_i] = exp( -deltaT / tau );
}

/**
* Remove a synapse from the network.
* @param neuron_i	Index of a neuron.
* @param syn_i		Index of a synapse.
*/
__device__ void removeSynapse( int neuron_i, int syn_i )
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
__device__ synapseType synType( neuronType* rgNeuronTypeMap_d, int ax, int ay, int bx, int by, int width )
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
__device__ int synSign( synapseType t )
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
