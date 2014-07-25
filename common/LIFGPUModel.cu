/** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **\ 
 * @authors Aaron Oziel, Sean Blackbourn 
 *
 * Fumitaka Kawasaki (5/3/14):
 * All functions were completed and working. Therefore, the followng comments
 * were removed. 
 *
 * Aaron Wrote (2/3/14):
 * All comments are now tracking progress in conversion from old GpuSim_struct.cu
 * file to the new one here. This is a quick key to keep track of their meanings. 
 *
 *	TODO = 	Needs work and/or is blank. Used to indicate possibly problematic 
 *				functions. 
 *	DONE = 	Likely complete functions. Will still need to be checked for
 *				variable continuity and proper arguments. 
 *   REMOVED =	Deleted, likely due to it becoming unnecessary or not necessary 
 *				for GPU implementation. These functions will likely have to be 
 *				removed from the LIFModel super class.
 *    COPIED = 	These functions were in the original GpuSim_struct.cu file 
 *				and were directly copy-pasted across to this file. 
 *
\** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **/

#include "LIFGPUModel.h"

extern "C" {
void normalMTGPU(float * randNoise_d);
void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_count);
}

#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif // PERFORMANCE_METRICS

//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllNeurons* allNeuronsDevice, AllSynapsesDevice* allSynapsesDevice );

//! Perform updating synapses for one time step.
__global__ void advanceSynapsesDevice ( int total_synapse_counts, LIFGPUModel::SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSynapsesDevice* allSynapsesDevice );

//! Calculate summation point.
__global__ void calcSummationMap( int totalNeurons, LIFGPUModel::SynapseIndexMap* synapseIndexMapDevice, AllSynapsesDevice* allSynapsesDevice );

//! Update the network.
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllNeurons* allNeuronsDevice, AllSynapsesDevice* allSynapsesDevice );

//! Add a synapse to the network.
__device__ void addSynapse( AllSynapsesDevice* allSynapsesDevice, synapseType type, const int src_neuron, const int dest_neuron, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons );

//! Create a synapse.
__device__ void createSynapse( AllSynapsesDevice* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type );

//! Remove a synapse from the network.
__device__ void eraseSynapse( AllSynapsesDevice* allSynapsesDevice, const int neuron_index, const int synapse_index, int maxSynapses );

//! Get the type of synapse.
__device__ synapseType synType( AllNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron );

//! Get the type of synapse (excitatory or inhibitory)
__device__ int synSign( synapseType t );

// ----------------------------------------------------------------------------

LIFGPUModel::LIFGPUModel() : 	
	allNeuronsDevice(NULL),
	allSynapsesDevice(NULL),
	synapseIndexMapDevice(NULL),
	randNoise_d(NULL)
{
}

LIFGPUModel::~LIFGPUModel() 
{
	//Let LIFModel base class handle de-allocation
}

/**
* Allocates memories on CUDA device.
* @param[in] sim_info			Pointer to the simulation information.
* @param[in] allNeuronsHost		List of all Neurons.
* @param[in] synapses			List of all Synapses.
*/
void LIFGPUModel::allocDeviceStruct(const SimulationInfo *sim_info, const AllNeurons &allNeuronsHost, AllSynapses &allSynapsesHost)
{
	// Allocate Neurons and Synapses strucs on GPU device memory
	int neuron_count = sim_info->totalNeurons;
	int max_synapses = sim_info->maxSynapsesPerNeuron;
	int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
	allocNeuronDeviceStruct( neuron_count, max_spikes );		// allocate device memory for each member
	allocSynapseDeviceStruct( neuron_count, max_synapses );	// allocate device memory for each member

	// Allocate memory for random noise array
	size_t randNoise_d_size = neuron_count * sizeof (float);	// size of random noise array
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );

	// Copy host neuron and synapse arrays into GPU device
	copyNeuronHostToDevice( allNeuronsHost, neuron_count );
	copySynapseHostToDevice( allSynapsesHost, sim_info );

	// allocate synapse inverse map
	allocSynapseImap( neuron_count );

	// create a synapse index map on device memory
	createSynapseImap(allSynapsesHost, sim_info);
}

/**
 *  Sets up the Simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  neurons     List of all Neurons.
 *  @param  synapses    List of all Synapses.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void LIFGPUModel::setupSim(const SimulationInfo *sim_info, const AllNeurons &neurons, AllSynapses &synapses, IRecorder* simRecorder)
{
    // Set device ID
    HANDLE_ERROR( cudaSetDevice( g_deviceId ) );

    LIFModel::setupSim(sim_info, neurons, synapses, simRecorder);
    allocDeviceStruct(sim_info, neurons, synapses);

    //initialize Mersenne Twister
    //assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
    int rng_blocks = 25; //# of blocks the kernel will use
    int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
    int rng_mt_rng_count = sim_info->totalNeurons/rng_nPerRng; //# of threads to generate for neuron_count rand #s
    int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
    initMTGPU(sim_info->seed, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

#ifdef PERFORMANCE_METRICS
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    t_gpu_rndGeneration = 0.0f;
    t_gpu_advanceNeurons = 0.0f;
    t_gpu_advanceSynapses = 0.0f;
    t_gpu_calcSummation = 0.0f;
#endif // PERFORMANCE_METRICS
}

/**
 *  Loads the simulation based on istream input.
 *  @param  input   istream to read from.
 *  @param  neurons list of neurons to set.
 *  @param  synapses    list of synapses to set.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void LIFGPUModel::loadMemory(istream& input, AllNeurons &allNeuronsHost, AllSynapses &allSynapsesHost, const SimulationInfo *sim_info)
{
    LIFModel::loadMemory(input, allNeuronsHost, allSynapsesHost, sim_info);
   
    // Reinitialize device struct - Copy host neuron and synapse arrays into GPU device
    int neuron_count = sim_info->totalNeurons;
    copyNeuronHostToDevice( allNeuronsHost, neuron_count );
    copySynapseHostToDevice( allSynapsesHost, sim_info );

    // create a synapse index map on device memory
    createSynapseImap(allSynapsesHost, sim_info);
}

/** 
*  Advance everything in the model one time step. In this case, that
*  means calling all of the kernels that do the "micro step" updating
*  (i.e., NOT the stuff associated with growth).
*  @param  neurons the Neuron list to search from.
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo class to read information from.
*/
void LIFGPUModel::advance(AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{
	int neuron_count = sim_info->totalNeurons;
	size_t total_synapse_counts = synapses.total_synapse_counts;

	// CUDA parameters
	const int threadsPerBlock = 256;
	int blocksPerGrid;

#ifdef PERFORMANCE_METRICS
	startTimer();
#endif // PERFORMANCE_METRICS

	normalMTGPU(randNoise_d);

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_rndGeneration);
#endif // PERFORMANCE_METRICS

	// display running info to console
	// Advance neurons ------------->
	blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

#ifdef PERFORMANCE_METRICS
	startTimer();
#endif // PERFORMANCE_METRICS

	advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, g_simulationStep, sim_info->maxSynapsesPerNeuron, sim_info->deltaT, randNoise_d, allNeuronsDevice, allSynapsesDevice );

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_advanceNeurons);
#endif // PERFORMANCE_METRICS

	// Advance synapses ------------->
	blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;

#ifdef PERFORMANCE_METRICS
	startTimer();
#endif // PERFORMANCE_METRICS

	advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, allSynapsesDevice );

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_advanceSynapses);
#endif // PERFORMANCE_METRICS

	// calculate summation point
	blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;

#ifdef PERFORMANCE_METRICS
	startTimer();
#endif // PERFORMANCE_METRICS

	calcSummationMap <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, synapseIndexMapDevice, allSynapsesDevice );

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_calcSummation);
#endif // PERFORMANCE_METRICS
}

/** 
*  Update the connection of all the Neurons and Synapses of the simulation.
*  @param  currentStep the current step of the simulation.
*  @param  neurons     the Neuron list to search from.
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo class to read information from.
*  @param  simRecorder Pointer to the simulation recordig object.
*/
void LIFGPUModel::updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info, IRecorder* simRecorder)
{
	updateHistory(currentStep, sim_info->epochDuration, neurons, sim_info, simRecorder);
	updateFrontiers(sim_info->totalNeurons);
	updateOverlap(sim_info->totalNeurons);
	updateWeights(sim_info->totalNeurons, neurons, synapses, sim_info);
}

/** 
*  Begin terminating the simulator.
*  @param  neurons     list of all Neurons.
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo to refer.
*/
void LIFGPUModel::cleanupSim(AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info)
{
    // copy device synapse and neuron structs to host memory
    copyNeuronDeviceToHost( neurons, sim_info->totalNeurons );
    copySynapseDeviceToHost( synapses, sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron );

    // Deallocate device memory
    deleteNeuronDeviceStruct(sim_info->totalNeurons);
    deleteSynapseDeviceStruct(sim_info->totalNeurons, sim_info->maxSynapsesPerNeuron);
    deleteSynapseImap();

    HANDLE_ERROR( cudaFree( randNoise_d ) );

#ifdef PERFORMANCE_METRICS
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
#endif // PERFORMANCE_METRICS
}

/* ------------------*\
|* # Helper Functions
\* ------------------*/

/**
 *  Allocate device memory for synapse inverse map.
 *  @param  count	The number of neurons.
 */
void LIFGPUModel::allocSynapseImap( int count )
{
	SynapseIndexMap synapseIndexMap;

	HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapse_begin, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.synapseCount, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMemset(synapseIndexMap.incomingSynapse_begin, 0, count * sizeof( int ) ) );
	HANDLE_ERROR( cudaMemset(synapseIndexMap.synapseCount, 0, count * sizeof( int ) ) );

	HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMapDevice, sizeof( SynapseIndexMap ) ) );
	HANDLE_ERROR( cudaMemcpy( synapseIndexMapDevice, &synapseIndexMap, sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );
}

/**
 *  Deallocate device memory for synapse inverse map.
 */
void LIFGPUModel::deleteSynapseImap(  )
{
	SynapseIndexMap synapseIndexMap;

	HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.incomingSynapse_begin ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.synapseCount ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.inverseIndex ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.activeSynapseIndex ) );
}

/** 
 *  Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
 *  @param  synapseIndexMapHost		Reference to the SynapseIndexMap in host memory.
 *  @param  neuron_count		The number of neurons.
 *  @param  total_synapse_counts	The number of synapses.
 */
void LIFGPUModel::copySynapseIndexMapHostToDevice(SynapseIndexMap &synapseIndexMapHost, int neuron_count, int total_synapse_counts)
{
	SynapseIndexMap synapseIndexMap;

	HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.incomingSynapse_begin, synapseIndexMapHost.incomingSynapse_begin, neuron_count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.synapseCount, synapseIndexMapHost.synapseCount, neuron_count * sizeof( int ), cudaMemcpyHostToDevice ) );
	// the number of synapses may change, so we reallocate the memory
	if (synapseIndexMap.inverseIndex != NULL) {
		HANDLE_ERROR( cudaFree( synapseIndexMap.inverseIndex ) );
	}
	HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.inverseIndex, total_synapse_counts * sizeof( uint32_t ) ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.inverseIndex, synapseIndexMapHost.inverseIndex, total_synapse_counts * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );

	if (synapseIndexMap.activeSynapseIndex != NULL) {
		HANDLE_ERROR( cudaFree( synapseIndexMap.activeSynapseIndex ) );
	}
	HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.activeSynapseIndex, total_synapse_counts * sizeof( uint32_t ) ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.activeSynapseIndex, synapseIndexMapHost.activeSynapseIndex, total_synapse_counts * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy ( synapseIndexMapDevice, &synapseIndexMap, sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );
}

/**
 *  Get synapse_counts in AllSynapses struct on device memory.
 *  @param  allSynapsesHost	Reference to the AllSynapses struct on host memory.
 *  @param  neuron_coun		The number of neurons.
 */
void LIFGPUModel::copyDeviceSynapseCountsToHost(AllSynapses &allSynapsesHost, int neuron_count)
{
	AllSynapsesDevice allSynapses;

	HANDLE_ERROR( cudaMemcpy ( &allSynapses, allSynapsesDevice, sizeof( AllSynapsesDevice ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.synapse_counts, allSynapses.synapse_counts, neuron_count * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
}

/** 
 *  Get summationCoord and in_use in AllSynapses struct on device memory.
 *  @param  allSynapsesHost     Reference to the AllSynapses struct on host memory.
 *  @param  neuron_coun         The number of neurons.
 *  @param  max_synapses	Maximum number of synapses per neuron.
 */
void LIFGPUModel::copyDeviceSynapseSumCoordToHost(AllSynapses &allSynapsesHost, int neuron_count, int max_synapses)
{
	AllSynapsesDevice allSynapses_0;
	AllSynapsesDevice allSynapses_1(neuron_count, max_synapses);

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, allSynapsesDevice, sizeof( AllSynapsesDevice ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.summationCoord, allSynapses_0.summationCoord,
                max_synapses * neuron_count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapses_1.in_use, allSynapses_0.in_use,
                max_synapses * neuron_count * sizeof( bool ), cudaMemcpyDeviceToHost ) );

	for (int i = 0; i < neuron_count; i++) {
		memcpy ( allSynapsesHost.summationCoord[i], &allSynapses_1.summationCoord[max_synapses * i],
                        max_synapses * sizeof( Coordinate ) );
		memcpy ( allSynapsesHost.in_use[i], &allSynapses_1.in_use[max_synapses * i],
                        max_synapses * sizeof( bool ) );
	}
}

/**
 *  Create a synapse index map on device memory.
 *  @param  synapses     Reference to the AllSynapses struct on host memory.
 *  @param] sim_info     Pointer to the simulation information.
 */
void LIFGPUModel::createSynapseImap(AllSynapses &synapses, const SimulationInfo* sim_info )
{
	int neuron_count = sim_info->totalNeurons;
	int width = sim_info->width;
	int total_synapse_counts = 0;

	// count the total synapses
        for ( int i = 0; i < neuron_count; i++ )
        {
                assert( synapses.synapse_counts[i] < synapses.max_synapses );
                total_synapse_counts += synapses.synapse_counts[i];
        }

        DEBUG ( cout << "total_synapse_counts: " << total_synapse_counts << endl; )

        if ( total_synapse_counts == 0 )
        {
                return;
        }

        // allocate memories for inverse map
        vector<uint32_t>* rgSynapseSynapseIndexMap = new vector<uint32_t>[neuron_count];

        uint32_t syn_i = 0;
	int n_inUse = 0;

        // create synapse inverse map
	SynapseIndexMap synapseIndexMap(neuron_count, total_synapse_counts);
        for (int i = 0; i < neuron_count; i++)
        {
                for ( int j = 0; j < synapses.max_synapses; j++, syn_i++ )
                {
                        if ( synapses.in_use[i][j] == true )
                        {
                                int idx = synapses.summationCoord[i][j].x
                                        + synapses.summationCoord[i][j].y * width;
                                rgSynapseSynapseIndexMap[idx].push_back(syn_i);

				synapseIndexMap.activeSynapseIndex[n_inUse] = syn_i;
                                n_inUse++;
                        }
                }
        }

        assert( total_synapse_counts == n_inUse ); 
        synapses.total_synapse_counts = total_synapse_counts; 

        syn_i = 0;
        for (int i = 0; i < neuron_count; i++)
        {
                synapseIndexMap.incomingSynapse_begin[i] = syn_i;
                synapseIndexMap.synapseCount[i] = rgSynapseSynapseIndexMap[i].size();

                for ( int j = 0; j < rgSynapseSynapseIndexMap[i].size(); j++, syn_i++)
                {
                        synapseIndexMap.inverseIndex[syn_i] = rgSynapseSynapseIndexMap[i][j];
                }
        }

        // copy inverse map to the device memory
	copySynapseIndexMapHostToDevice(synapseIndexMap, neuron_count, total_synapse_counts);

        // delete memories
        delete[] rgSynapseSynapseIndexMap;
}

/**
 *  Get spike history in AllNeurons struct on device memory.
 *  @param  allNeuronsHost      Reference to the allNeurons struct on host memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void LIFGPUModel::copyDeviceSpikeHistoryToHost(AllNeurons &allNeuronsHost, const SimulationInfo *sim_info)
{
	AllNeurons allNeurons;
	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllNeurons ), cudaMemcpyDeviceToHost ) );

	int numNeurons = sim_info->totalNeurons;
	uint64_t* pSpikeHistory[numNeurons];
	HANDLE_ERROR( cudaMemcpy ( pSpikeHistory, allNeurons.spike_history, numNeurons * sizeof( uint64_t* ), cudaMemcpyDeviceToHost ) );

	int max_spikes = static_cast<int> (sim_info->epochDuration * sim_info->maxFiringRate);
	for (int i = 0; i < numNeurons; i++) {
		HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.spike_history[i], pSpikeHistory[i], 
			max_spikes * sizeof( uint64_t ), cudaMemcpyDeviceToHost ) );
	}
}

/**
 *  Get spikeCount in AllNeurons struct on device memory.
 *  @param  allNeuronsHost      Reference to the allNeurons struct on host memory.
 *  @param  numNeurons          The number of neurons.
 */
void LIFGPUModel::copyDeviceSpikeCountsToHost(AllNeurons &allNeuronsHost, int numNeurons)
{
	AllNeurons allNeurons;
	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllNeurons ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.spikeCount, allNeurons.spikeCount, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/** 
*  Clear the spike counts out of all Neurons.
*  @param  numNeurons The number of neurons.
*/
void LIFGPUModel::clearSpikeCounts(int numNeurons)
{
	AllNeurons allNeurons;
	HANDLE_ERROR( cudaMemcpy ( &allNeurons, allNeuronsDevice, sizeof( AllNeurons ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemset( allNeurons.spikeCount, 0, numNeurons * sizeof( int ) ) );
}

/**
 *  Update the Neuron's history.
 *  @param  currentStep current step of the simulation
 *  @param  epochDuration    duration of the 
 *  @param  neurons the list to update.
 *  @param  sim_info    SimulationInfo to refer from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void LIFGPUModel::updateHistory(const int currentStep, BGFLOAT epochDuration, AllNeurons &neurons, const SimulationInfo *sim_info, IRecorder* simRecorder)
{
    // Calculate growth cycle firing rate for previous period
    copyDeviceSpikeCountsToHost(neurons, sim_info->totalNeurons);
    copyDeviceSpikeHistoryToHost(neurons, sim_info);

    LIFModel::updateHistory(currentStep, epochDuration, neurons, sim_info, simRecorder);

    // clear spike count
    clearSpikeCounts(sim_info->totalNeurons);
}

/** 
*  Update the weight of the Synapses in the simulation.
*  Note: Platform Dependent.
*  @param  num_neurons number of neurons to update.
*  @param  neurons the Neuron list to search from.
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo to refer from.
*/
void LIFGPUModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{
	// For now, we just set the weights to equal the areas. We will later
	// scale it and set its sign (when we index and get its sign).
	m_conns->W = m_conns->area;

	int width = sim_info->width;
	BGFLOAT deltaT = sim_info->deltaT;

	// CUDA parameters
	const int threadsPerBlock = 256;
	int blocksPerGrid;

	// allocate memories
	size_t W_d_size = sim_info->totalNeurons * sim_info->totalNeurons * sizeof (BGFLOAT);
	BGFLOAT* W_h = new BGFLOAT[W_d_size];
	BGFLOAT* W_d;
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

	// copy weight data to the device memory
	for ( int i = 0 ; i < sim_info->totalNeurons; i++ )
		for ( int j = 0; j < sim_info->totalNeurons; j++ )
			W_h[i * sim_info->totalNeurons + j] = m_conns->W(i, j);

	HANDLE_ERROR( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

	blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
	updateNetworkDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, width, deltaT, W_d, sim_info->maxSynapsesPerNeuron, allNeuronsDevice, allSynapsesDevice );

	// free memories
	HANDLE_ERROR( cudaFree( W_d ) );
	delete[] W_h;

        // copy device synapse count to host memory
	copyDeviceSynapseCountsToHost(synapses, num_neurons);
        // copy device synapse summation coordinate to host memory
	copyDeviceSynapseSumCoordToHost(synapses, num_neurons, sim_info->maxSynapsesPerNeuron);
	// create synapse inverse map
	createSynapseImap( synapses, sim_info );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

// CUDA code for advancing neurons
/**
* @param[in] totalNeurons	Number of neurons.
* @param[in] simulationStep	The current simulation step.
* @param[in] maxSynapses	Maximum number of synapses per neuron.
* @param[in] deltaT		Inner simulation step duration.
* @param[in] randNoise		Pointer to device random noise array.
* @param[in] allNeuronsDevice	Pointer to Neuron structures in device memory.
* @param[in] allSynapsesDevice	Pointer to Synapse structures in device memory.
*/
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllNeurons* allNeuronsDevice, AllSynapsesDevice* allSynapsesDevice ) {
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



/** 
* @param[in] total_synapse_counts	Total number of synapses.
* @param[in] synapseIndexMap		Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
* @param[in] simulationStep		The current simulation step.
* @param[in] deltaT			Inner simulation step duration.
* @param[in] allSynapsesDevice	Pointer to Synapse structures in device memory.
*/
__global__ void advanceSynapsesDevice ( int total_synapse_counts, LIFGPUModel::SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllSynapsesDevice* allSynapsesDevice ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= total_synapse_counts )
		return;

	uint32_t iSyn = synapseIndexMapDevice->activeSynapseIndex[idx];

	BGFLOAT &psr = allSynapsesDevice->psr[iSyn];
	BGFLOAT decay = allSynapsesDevice->decay[iSyn];

	// Checks if there is an input spike in the queue.
	uint32_t &delay_queue = allSynapsesDevice->delayQueue[iSyn];
	int &delayIdx = allSynapsesDevice->delayIdx[iSyn];
	int ldelayQueue = allSynapsesDevice->ldelayQueue[iSyn];

	uint32_t delayMask = (0x1 << delayIdx);
	bool isFired = delay_queue & (delayMask);
	delay_queue &= ~(delayMask);
	if ( ++delayIdx >= ldelayQueue ) {
		delayIdx = 0;
	}

	// is an input in the queue?
	if (isFired) {
		uint64_t &lastSpike = allSynapsesDevice->lastSpike[iSyn];
		BGFLOAT &r = allSynapsesDevice->r[iSyn];
		BGFLOAT &u = allSynapsesDevice->u[iSyn];
		BGFLOAT D = allSynapsesDevice->D[iSyn];
		BGFLOAT F = allSynapsesDevice->F[iSyn];
		BGFLOAT U = allSynapsesDevice->U[iSyn];
		BGFLOAT W = allSynapsesDevice->W[iSyn];

		// adjust synapse parameters
		if (lastSpike != ULONG_MAX) {
			BGFLOAT isi = (simulationStep - lastSpike) * deltaT ;
			r = 1 + ( r * ( 1 - u ) - 1 ) * exp( -isi / D );
			u = U + u * ( 1 - U ) * exp( -isi / F );
		}
		psr += ( ( W / decay ) * u * r );// calculate psr
		lastSpike = simulationStep; // record the time of the spike
	}

	// decay the post spike response
	psr *= decay;
}

/** 
* @param[in] totalNeurons	Number of neurons.
* @param[in] synapseIndexMap	Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void calcSummationMap( int totalNeurons, LIFGPUModel::SynapseIndexMap* synapseIndexMapDevice, AllSynapsesDevice* allSynapsesDevice ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= totalNeurons )
		return;
	
	uint32_t iCount = synapseIndexMapDevice->synapseCount[idx];
	if (iCount != 0) {
		int beginIndex = synapseIndexMapDevice->incomingSynapse_begin[idx];
		uint32_t* inverseMap_begin = &( synapseIndexMapDevice->inverseIndex[beginIndex] );
		BGFLOAT sum = 0.0;
		uint32_t syn_i = inverseMap_begin[0];
		BGFLOAT &summationPoint = *( allSynapsesDevice->summationPoint[syn_i] );
		for ( uint32_t i = 0; i < iCount; i++ ) {
			syn_i = inverseMap_begin[i];
			sum += allSynapsesDevice->psr[syn_i];
		}
		summationPoint = sum;
	}
} 

/** 
* Adjust the strength of the synapse or remove it from the synapse map if it has gone below 
* zero.
* @param[in] num_neurons	Number of neurons.
* @param[in] width		Width of neuron map (assumes square).
* @param[in] deltaT		The time step size.
* @param[in] W_d		Array of synapse weight.
* @param[in] maxSynapses	Maximum number of synapses per neuron.
* @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
* @param[in] allSynapsesDevice         Pointer to the Synapse structures in device memory.
*/
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllNeurons* allNeuronsDevice, AllSynapsesDevice* allSynapsesDevice )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_neurons )
        return;

    int adjusted = 0;
    //int could_have_been_removed = 0; // TODO: use this value
    int removed = 0;
    int added = 0;

    // Scale and add sign to the areas
    // visit each neuron 'a'
    int src_neuron = idx;
    int xa = src_neuron % width;
    int ya = src_neuron / width;

    // and each destination neuron 'b'
    for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
        int xb = dest_neuron % width;
        int yb = dest_neuron / width;

        // visit each synapse at (xa,ya)
        bool connected = false;
        synapseType type = synType(allNeuronsDevice, src_neuron, dest_neuron);

        // for each existing synapse
        size_t synapse_counts = allSynapsesDevice->synapse_counts[src_neuron];
        int synapse_adjusted = 0;
        for (size_t synapse_index = 0; synapse_adjusted < synapse_counts; synapse_index++) {
            uint32_t iSyn = maxSynapses * src_neuron + synapse_index; 
            if (allSynapsesDevice->in_use[iSyn] == true) {
                // if there is a synapse between a and b
                if (allSynapsesDevice->summationCoord[iSyn].x == xb &&
                    allSynapsesDevice->summationCoord[iSyn].y == yb) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.
                    if (W_d[src_neuron * num_neurons + dest_neuron] < 0) {
                        removed++;
                        eraseSynapse(allSynapsesDevice, src_neuron, synapse_index, maxSynapses);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesDevice->W[iSyn] = W_d[src_neuron * num_neurons 
                            + dest_neuron] * synSign(type) * SYNAPSE_STRENGTH_ADJUSTMENT;
                    }
                }
                synapse_adjusted++;
            }
        }

        // if not connected and weight(a,b) > 0, add a new synapse from a to b
        if (!connected && (W_d[src_neuron * num_neurons +  dest_neuron] > 0)) {
            // locate summation point
            BGFLOAT* sum_point = &( allNeuronsDevice->summation_map[dest_neuron] );
            added++;

            addSynapse(allSynapsesDevice, type, src_neuron, dest_neuron, xa, ya, xb, yb, sum_point, deltaT, W_d, num_neurons);

        }
    } 
}

/* ------------------*\
|* # Device Functions
\* ------------------*/

/** 
* Remove a synapse from the network.
* @param[in] allSynapsesDevice         Pointer to the Synapse structures in device memory.
* @param neuron_index	Index of a neuron.
* @param synapse_index	Index of a synapse.
* @param[in] maxSynapses	Maximum number of synapses per neuron.
*/
__device__ void eraseSynapse( AllSynapsesDevice* allSynapsesDevice, const int neuron_index, const int synapse_index, int maxSynapses )
{
    uint32_t iSync = maxSynapses * neuron_index + synapse_index;
    allSynapsesDevice->synapse_counts[neuron_index]--;
    allSynapsesDevice->in_use[iSync] = false;
    allSynapsesDevice->summationPoint[iSync] = NULL;
}

/** 
* Adds a synapse to the network.  Requires the locations of the source and
* destination neurons.
* @param allSynapsesDevice      Pointer to the Synapse structures in device memory.
* @param type    		Type of the Synapse to create.
* @param src_neuron		Index of the source neuron.
* @param dest_neuron		Index of the destination neuron.
* @param source_x		X location of source.
* @param source_y		Y location of source.
* @param dest_x			X location of destination.
* @param dest_y			Y location of destination.
* @param sum_point		Pointer to the summation point.
* @param deltaT			The time step size.
* @param W_d			Array of synapse weight.
* @param num_neurons		The number of neurons.
*/
__device__ void addSynapse(AllSynapsesDevice* allSynapsesDevice, synapseType type, const int src_neuron, const int dest_neuron, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons)
{
    if (allSynapsesDevice->synapse_counts[src_neuron] >= allSynapsesDevice->max_synapses) {
        return; // TODO: ERROR!
    }

    // add it to the list
    size_t synapse_index;
    size_t max_synapses = allSynapsesDevice->max_synapses;
    uint32_t iSync = max_synapses * src_neuron;
    for (synapse_index = 0; synapse_index < max_synapses; synapse_index++) {
        if (!allSynapsesDevice->in_use[iSync + synapse_index]) {
            break;
        }
    }

    allSynapsesDevice->synapse_counts[src_neuron]++;

    // create a synapse
    createSynapse(allSynapsesDevice, src_neuron, synapse_index, source_x, source_y, dest_x, dest_y, sum_point, deltaT, type );
    allSynapsesDevice->W[iSync + synapse_index] = W_d[src_neuron * num_neurons + dest_neuron] * synSign(type) * SYNAPSE_STRENGTH_ADJUSTMENT;
}

/**
 *  Create a Synapse and connect it to the model.
 *  @param allSynapsesDevice    Pointer to the Synapse structures in device memory.
 *  @param neuron_index    	Index of the source neuron.
 *  @param synapse_index   	Index of the Synapse to create.
 *  @param source_x             X location of source.
 *  @param source_y             Y location of source.
 *  @param dest_x               X location of destination.
 *  @param dest_y               Y location of destination.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createSynapse(AllSynapsesDevice* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    size_t max_synapses = allSynapsesDevice->max_synapses;
    uint32_t iSyn = max_synapses * neuron_index + synapse_index;

    allSynapsesDevice->in_use[iSyn] = true;
    allSynapsesDevice->summationPoint[iSyn] = sum_point;
    allSynapsesDevice->summationCoord[iSyn].x = dest_x;
    allSynapsesDevice->summationCoord[iSyn].y = dest_y;
    allSynapsesDevice->synapseCoord[iSyn].x = source_x;
    allSynapsesDevice->synapseCoord[iSyn].y = source_y;
    allSynapsesDevice->W[iSyn] = 10.0e-9;

    allSynapsesDevice->delayQueue[iSyn] = 0;
    allSynapsesDevice->delayIdx[iSyn] = 0;
    allSynapsesDevice->ldelayQueue[iSyn] = LENGTH_OF_DELAYQUEUE;

    allSynapsesDevice->psr[iSyn] = 0.0;
    allSynapsesDevice->r[iSyn] = 1.0;
    allSynapsesDevice->u[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesDevice->lastSpike[iSyn] = ULONG_MAX;
    allSynapsesDevice->type[iSyn] = type;

    allSynapsesDevice->U[iSyn] = DEFAULT_U;
    allSynapsesDevice->tau[iSyn] = DEFAULT_tau;

    BGFLOAT U;
    BGFLOAT D;
    BGFLOAT F;
    BGFLOAT tau;
    switch (type) {
        case II:
            U = 0.32;
            D = 0.144;
            F = 0.06;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            U = 0.25;
            D = 0.7;
            F = 0.02;
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            U = 0.05;
            D = 0.125;
            F = 1.2;
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            U = 0.5;
            D = 1.1;
            F = 0.05;
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesDevice->U[iSyn] = U;
    allSynapsesDevice->D[iSyn] = D;
    allSynapsesDevice->F[iSyn] = F;

    allSynapsesDevice->tau[iSyn] = tau;
    allSynapsesDevice->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesDevice->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    size_t size = allSynapsesDevice->total_delay[iSyn] / ( sizeof(uint8_t) * 8 ) + 1;
    assert( size <= BYTES_OF_DELAYQUEUE );
}

/** 
* Returns the type of synapse at the given coordinates
* @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
* @param src_neuron		Index of the source neuron.
* @param dest_neuron		Index of the destination neuron.
*/
__device__ synapseType synType( AllNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron )
{
    if ( allNeuronsDevice->neuron_type_map[src_neuron] == INH && allNeuronsDevice->neuron_type_map[dest_neuron] == INH )
        return II;
    else if ( allNeuronsDevice->neuron_type_map[src_neuron] == INH && allNeuronsDevice->neuron_type_map[dest_neuron] == EXC )
        return IE;
    else if ( allNeuronsDevice->neuron_type_map[src_neuron] == EXC && allNeuronsDevice->neuron_type_map[dest_neuron] == INH )
        return EI;
    else if ( allNeuronsDevice->neuron_type_map[src_neuron] == EXC && allNeuronsDevice->neuron_type_map[dest_neuron] == EXC )
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
