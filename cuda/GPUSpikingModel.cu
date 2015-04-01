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
 *				removed from the Model super class.
 *    COPIED = 	These functions were in the original GpuSim_struct.cu file 
 *				and were directly copy-pasted across to this file. 
 *
\** - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **/

#include "GPUSpikingModel.h"

extern "C" {
void normalMTGPU(float * randNoise_d);
void initMTGPU(unsigned int seed, unsigned int blocks, unsigned int threads, unsigned int nPerRng, unsigned int mt_rng_count);
}

#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif // PERFORMANCE_METRICS

// ----------------------------------------------------------------------------

GPUSpikingModel::GPUSpikingModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 	
	Model::Model(conns, neurons, synapses, layout),
	synapseIndexMapDevice(NULL),
	randNoise_d(NULL)
{
}

GPUSpikingModel::~GPUSpikingModel() 
{
	//Let Model base class handle de-allocation
}

/**
* Allocates memories on CUDA device.
* @param[in] sim_info			Pointer to the simulation information.
*/
void GPUSpikingModel::allocDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info)
{
	// Allocate Neurons and Synapses strucs on GPU device memory
	m_neurons->allocNeuronDeviceStruct( allNeuronsDevice, sim_info );
	m_synapses->allocSynapseDeviceStruct( allSynapsesDevice, sim_info );

	// Allocate memory for random noise array
	int neuron_count = sim_info->totalNeurons;
	size_t randNoise_d_size = neuron_count * sizeof (float);	// size of random noise array
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );

	// Copy host neuron and synapse arrays into GPU device
	m_neurons->copyNeuronHostToDevice( *allNeuronsDevice, sim_info );
	m_synapses->copySynapseHostToDevice( *allSynapsesDevice, sim_info );

	// allocate synapse inverse map
	allocSynapseImap( neuron_count );

	// create a synapse index map on device memory
	createSynapseImap(*m_synapses, sim_info);
}

/**
 *  Sets up the Simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void GPUSpikingModel::setupSim(SimulationInfo *sim_info, IRecorder* simRecorder)
{
    // Set device ID
    HANDLE_ERROR( cudaSetDevice( g_deviceId ) );

    Model::setupSim(sim_info, simRecorder);

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
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void GPUSpikingModel::loadMemory(istream& input, const SimulationInfo *sim_info)
{
    Model::loadMemory(input, sim_info);
   
    // create a synapse index map on device memory
    createSynapseImap(*m_synapses, sim_info);
}

/** 
*  Advance everything in the model one time step. In this case, that
*  means calling all of the kernels that do the "micro step" updating
*  (i.e., NOT the stuff associated with growth).
*  @param  sim_info    SimulationInfo class to read information from.
*/
void GPUSpikingModel::advance(const SimulationInfo *sim_info)
{
	size_t total_synapse_counts = m_synapses->total_synapse_counts;

	// CUDA parameters
	const int threadsPerBlock = 256;
	int blocksPerGrid;

#ifdef PERFORMANCE_METRICS
	startTimer();
#endif // PERFORMANCE_METRICS

	normalMTGPU(randNoise_d);

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_rndGeneration);
	startTimer();
#endif // PERFORMANCE_METRICS

	// display running info to console
	// Advance neurons ------------->
	advanceNeurons(sim_info);

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_advanceNeurons);
	startTimer();
#endif // PERFORMANCE_METRICS

	// Advance synapses ------------->
	advanceSynapses(sim_info);

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_advanceSynapses);
	startTimer();
#endif // PERFORMANCE_METRICS

	// calculate summation point
        calcSummationMap(sim_info);

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_calcSummation);
#endif // PERFORMANCE_METRICS
}

/** 
*  Update the connection of all the Neurons and Synapses of the simulation.
*  @param  currentStep the current step of the simulation.
*  @param  sim_info    SimulationInfo class to read information from.
*  @param  simRecorder Pointer to the simulation recordig object.
*/
void GPUSpikingModel::updateConnections(const int currentStep, const SimulationInfo *sim_info, IRecorder* simRecorder)
{
	const int num_neurons = sim_info->totalNeurons;
	updateHistory(currentStep, sim_info->epochDuration, *m_neurons, sim_info, simRecorder);
	// Update the distance between frontiers of Neurons
	m_conns->updateFrontiers(num_neurons);
	// Update the areas of overlap in between Neurons
	m_conns->updateOverlap(num_neurons);
	updateWeights(sim_info->totalNeurons, *m_neurons, *m_synapses, sim_info);
}

/** 
*  Begin terminating the simulator.
*  @param  sim_info    SimulationInfo to refer.
*/
void GPUSpikingModel::cleanupSim(SimulationInfo *sim_info)
{
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
void GPUSpikingModel::allocSynapseImap( int count )
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
void GPUSpikingModel::deleteSynapseImap(  )
{
	SynapseIndexMap synapseIndexMap;

	HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.incomingSynapse_begin ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.synapseCount ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.inverseIndex ) );
	HANDLE_ERROR( cudaFree( synapseIndexMap.activeSynapseIndex ) );
	HANDLE_ERROR( cudaFree( synapseIndexMapDevice ) );
}

/** 
 *  Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
 *  @param  synapseIndexMapHost		Reference to the SynapseIndexMap in host memory.
 *  @param  neuron_count		The number of neurons.
 *  @param  total_synapse_counts	The number of synapses.
 */
void GPUSpikingModel::copySynapseIndexMapHostToDevice(SynapseIndexMap &synapseIndexMapHost, int neuron_count, int total_synapse_counts)
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
 *  Create a synapse index map on device memory.
 *  @param  synapses     Reference to the AllSynapses struct on host memory.
 *  @param] sim_info     Pointer to the simulation information.
 */
void GPUSpikingModel::createSynapseImap(AllSynapses &synapses, const SimulationInfo* sim_info )
{
	int neuron_count = sim_info->totalNeurons;
	int width = sim_info->width;
	int total_synapse_counts = 0;

	// count the total synapses
        for ( int i = 0; i < neuron_count; i++ )
        {
                assert( synapses.synapse_counts[i] < synapses.maxSynapsesPerNeuron );
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
                for ( int j = 0; j < synapses.maxSynapsesPerNeuron; j++, syn_i++ )
                {
                        uint32_t iSyn = synapses.maxSynapsesPerNeuron * i + j;
                        if ( synapses.in_use[iSyn] == true )
                        {
                                int idx = synapses.summationCoord[iSyn].x
                                        + synapses.summationCoord[iSyn].y * width;
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
 *  Update the Neuron's history.
 *  @param  currentStep current step of the simulation
 *  @param  epochDuration    duration of the 
 *  @param  neurons the list to update.
 *  @param  sim_info    SimulationInfo to refer from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void GPUSpikingModel::updateHistory(const int currentStep, BGFLOAT epochDuration, AllNeurons &neurons, const SimulationInfo *sim_info, IRecorder* simRecorder)
{
    // Calculate growth cycle firing rate for previous period
    AllSpikingNeurons &spikingNeurons = dynamic_cast<AllSpikingNeurons &>(neurons);
    copyDeviceSpikeCountsToHost(spikingNeurons, sim_info->totalNeurons);
    copyDeviceSpikeHistoryToHost(spikingNeurons, sim_info);

    Model::updateHistory(currentStep, epochDuration, sim_info, simRecorder);

    // clear spike count
    clearSpikeCounts(sim_info->totalNeurons);
}

