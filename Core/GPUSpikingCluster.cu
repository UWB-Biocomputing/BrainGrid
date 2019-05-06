/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **\ 
 * @authors Aaron Oziel, Sean Blackbourn 
 *
 * Fumitaka Kawasaki (1/27/17):
 * Changed from Model to Cluster class.
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

#include "GPUSpikingCluster.h"

GPUSpikingCluster::GPUSpikingCluster(IAllNeurons *neurons, IAllSynapses *synapses) : 	
	Cluster::Cluster(neurons, synapses),
	m_synapseIndexMapDevice(NULL),
	randNoise_d(NULL),
	m_allNeuronsDevice(NULL),
	m_allSynapsesDevice(NULL)
{
}

/*
 * Allocates and initializes memories on CUDA device.
 *
 * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
 * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
 * @param[in]  sim_info			Pointer to the simulation information.
 * @param[in]  clr_info			Pointer to the cluste information.
 */
void GPUSpikingCluster::allocDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info, ClusterInfo *clr_info)
{
	DEBUG({reportGPUMemoryUsage(clr_info);})

	// Allocate Neurons and Synapses strucs on GPU device memory
	m_neurons->allocNeuronDeviceStruct( allNeuronsDevice, sim_info, clr_info );
	m_synapses->allocSynapseDeviceStruct( allSynapsesDevice, sim_info, clr_info );

	// Copy host neuron and synapse arrays into GPU device
	m_neurons->copyNeuronHostToDevice( *allNeuronsDevice, sim_info, clr_info );
	m_synapses->copySynapseHostToDevice( *allSynapsesDevice, sim_info, clr_info );

	// allocate synapse index map in device memory
	int neuron_count = clr_info->totalClusterNeurons;
	allocSynapseImap( neuron_count );

	#if defined(VALIDATION)
		// allocate buffer to save random numbers in host memory
		if (clr_info->clusterID == 0) {
			BGSIZE randNoiseBufferSize = sizeof(float) * sim_info->totalNeurons * sim_info->minSynapticTransDelay;
			m_randNoiseHost = new float[randNoiseBufferSize];
		}
	#endif

	DEBUG({reportGPUMemoryUsage(clr_info);})
}

void GPUSpikingCluster::reportGPUMemoryUsage(const ClusterInfo *clr_info)
{
	printf("\n");

	size_t free_byte;
	size_t total_byte;

	checkCudaErrors( cudaMemGetInfo( &free_byte, &total_byte ) );

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	printf("After allocating GPU memories\n");
	printf("GPU memory usage: device ID = %d, used = %5.3f MB, free = %5.3f MB, total = %5.3f MB\n",
				clr_info->deviceId, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

	printf("\n");
}

/*
 * Copies device memories to host memories and deallocaes them.
 *
 * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
 * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
 * @param[in]  sim_info                  Pointer to the simulation information.
 * @param[in]  clr_info                  Pointer to the cluster information.
 */
void GPUSpikingCluster::deleteDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info, ClusterInfo *clr_info) 
{
    // copy device synapse and neuron structs to host memory
    m_neurons->copyNeuronDeviceToHost( *allNeuronsDevice, sim_info, clr_info );

    // Deallocate device memory
    m_neurons->deleteNeuronDeviceStruct( *allNeuronsDevice, clr_info );

    // copy device synapse and neuron structs to host memory
    m_synapses->copySynapseDeviceToHost( *allSynapsesDevice, sim_info, clr_info );

    // Deallocate device memory
    m_synapses->deleteSynapseDeviceStruct( *allSynapsesDevice );

    deleteSynapseImap();

    #if defined(VALIDATION)
		// Deallocate buffer to save random numbers in host memory
		if (clr_info->clusterID == 0) delete[] m_randNoiseHost;
    #endif

    checkCudaErrors( cudaFree( randNoise_d ) );
}

/*
 *  Sets up the Simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  clr_info    ClusterInfo class to read information from.
 */
void GPUSpikingCluster::setupCluster(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info)
{
	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );
	Cluster::setupCluster(sim_info, layout, clr_info);

	// initialize Mersenne Twister
	#if defined(VALIDATION)
		if (clr_info->clusterID == 0) {
			//assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
			int rng_blocks = 25; //# of blocks the kernel will use
			int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
			int rng_mt_rng_count = sim_info->totalNeurons / rng_nPerRng; //# of threads to generate for neuron_count rand #s
			int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed

			initMTGPU(clr_info->seed, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

			// Allocate memory for random noise array
			BGSIZE randNoise_d_size = rng_mt_rng_count * rng_nPerRng * sizeof (float) * sim_info->minSynapticTransDelay;	// size of random noise array
			checkCudaErrors( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );
		} else {
			BGSIZE randNoise_d_size = clr_info->totalClusterNeurons * sizeof (float);	// size of random noise array
			checkCudaErrors( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );
		}
	#else // !VALIDATION

		// rng_mt_rng_count is the number of threads to be created in Mersenne Twister,
		// maximum number of which is defined by MT_RNG_COUNT.
		// rng_mt_rng_count must be a multiple of the number of warp for coaleased write.
		// rng_nPerRng is the thread granularity. Therefore, the number of random numbers 
		// generated is (rng_mt_rng_count * rng_nPerRng).
		// rng_nPerRn must be even.
		// Here we find a minimum rng_nPerRng that satisfies 
		// (rng_mt_rng_count * rng_nPerRng) >= neuron_count.

		int neuron_count = clr_info->totalClusterNeurons;  // # of total neurons in the cluster
		int rng_threads = 128;  // # of threads per block (must be a multiple of # of warp for coaleased write)
		int rng_nPerRng = 0;    // # of iterations per thread (thread granularity, # of rands generated per thread, must be even)
		int rng_mt_rng_count;   // # of threads to generate for neuron_count rand #s
		int rng_blocks;         // # of blocks the kernel will use

		do {
			rng_nPerRng += 2;
			rng_mt_rng_count = (neuron_count - 1) / rng_nPerRng + 1; 
			rng_blocks = (rng_mt_rng_count + rng_threads - 1) / rng_threads; 
			rng_mt_rng_count = rng_threads * rng_blocks; // # of threads must be a multiple of # of threads per block
		} while (rng_mt_rng_count > MT_RNG_COUNT);     // rng_mt_rng_count must be <= MT_RNG_COUNT

		initMTGPU(clr_info->seed, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

		// Allocate memory for random noise array
		BGSIZE randNoise_d_size = rng_mt_rng_count * rng_nPerRng * sizeof (float);	// size of random noise array
		checkCudaErrors( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );
	#endif // VALIDATION

	#ifdef PERFORMANCE_METRICS
		cudaEventCreate( &clr_info->start );
		cudaEventCreate( &clr_info->stop );

		clr_info->t_gpu_rndGeneration = 0.0;
		clr_info->t_gpu_advanceNeurons = 0.0;
		clr_info->t_gpu_advanceSynapses = 0.0;
		clr_info->t_gpu_calcSummation = 0.0;
		clr_info->t_gpu_updateConns = 0.0;
		clr_info->t_gpu_setupConns = 0.0;
		clr_info->t_gpu_updateSynapsesWeights = 0.0;
		clr_info->t_gpu_processInterClustesOutgoingSpikes = 0.0;
		clr_info->t_gpu_processInterClustesIncomingSpikes = 0.0;
	#endif // PERFORMANCE_METRICS

	allocDeviceStruct((void **)&m_allNeuronsDevice, (void **)&m_allSynapsesDevice, sim_info, clr_info);

	// set some parameters used for advanceNeuronsDevice and advanceSynapsesDevice
	m_neurons->setAdvanceNeuronsDeviceParams(*m_synapses);
	m_synapses->setAdvanceSynapsesDeviceParams();

	//clr_info->fpCalcSummationMap = &GPUSpikingCluster::calcSummationMap;
}

/* 
 *  Begin terminating the simulator.
 *
 *  @param  sim_info    SimulationInfo to refer.
 *  @param  clr_info    ClusterInfo to refer.
 */
void GPUSpikingCluster::cleanupCluster(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );
	deleteDeviceStruct((void**)&m_allNeuronsDevice, (void**)&m_allSynapsesDevice, sim_info, clr_info);

	#ifdef PERFORMANCE_METRICS
		cudaEventDestroy( clr_info->start );
		cudaEventDestroy( clr_info->stop );
	#endif

	Cluster::cleanupCluster(sim_info, clr_info);
}

/*
 *  Loads the simulation based on istream input.
 *
 *  @param  input   istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 *  @param  clr_info  ClusterInfo to refer.
 */
void GPUSpikingCluster::deserialize(istream& input, const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
	Cluster::deserialize(input, sim_info, clr_info);

	// Reinitialize device struct - Copy host neuron and synapse arrays into GPU device
	m_neurons->copyNeuronHostToDevice( m_allNeuronsDevice, sim_info, clr_info );
	m_synapses->copySynapseHostToDevice( m_allSynapsesDevice, sim_info, clr_info );
}

#if defined(VALIDATION)
/*
 *  Generates random numbers.
 *
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 *  @param  clr_info  ClusterInfo to refer.
 */
void GPUSpikingCluster::genRandNumbers(const SimulationInfo *sim_info, ClusterInfo *clr_info)
{
	// generates random numbers only in cluster 0
	if (clr_info->clusterID != 0) return;

	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

	#ifdef PERFORMANCE_METRICS
		cudaStartTimer(clr_info);
	#endif

	// generates random numbers for all clusters within transmission delay
	for (int i = 0; i < m_nSynapticTransDelay; i++) 
		normalMTGPU(randNoise_d + sim_info->totalNeurons * i);

	// and copy them to host memory to share among clusters
	BGSIZE randNoiseBufferSize = sizeof (float) * sim_info->totalNeurons * m_nSynapticTransDelay;
	checkCudaErrors( cudaMemcpy ( m_randNoiseHost, randNoise_d, randNoiseBufferSize,  cudaMemcpyDeviceToHost ) );

	#ifdef PERFORMANCE_METRICS
		cudaLapTime(clr_info, clr_info->t_gpu_rndGeneration);
	#endif
}
#endif

/*
 * Advances neurons network state of the cluster one simulation step.
 *
 * @param sim_info   parameters defining the simulation to be run with
 *                   the given collection of neurons.
 * @param clr_info   ClusterInfo to refer.
 * @param iStepOffset  Offset from the current simulation step.
 */
void GPUSpikingCluster::advanceNeurons(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset)
{
	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

	#ifdef PERFORMANCE_METRICS
		cudaStartTimer(clr_info);
	#endif

	#if defined(VALIDATION)
		// Get an appropriate pointer to the buffer of random numbers
		// Random numbers are stored in device memory of cluster 0, 
		// or in host memory for other clusters.
		float *randNoiseDevice;
		if (clr_info->clusterID == 0) {
			randNoiseDevice = randNoise_d + clr_info->totalClusterNeurons * (iStepOffset * sim_info->numClusters);
		} else {
			float *randNoiseHost = m_randNoiseHost + clr_info->totalClusterNeurons * (iStepOffset * sim_info->numClusters + clr_info->clusterID);
			checkCudaErrors( cudaMemcpy( randNoise_d, randNoiseHost, sizeof( float ) * clr_info->totalClusterNeurons, cudaMemcpyHostToDevice ) );
			randNoiseDevice = randNoise_d;
		}
	#else
		normalMTGPU(randNoise_d);

		#ifdef PERFORMANCE_METRICS
			cudaLapTime(clr_info, clr_info->t_gpu_rndGeneration);
			cudaStartTimer(clr_info);
		#endif
	#endif

	m_neurons->advanceNeurons(*m_synapses, m_allNeuronsDevice, m_allSynapsesDevice, sim_info, randNoise_d, m_synapseIndexMapDevice, clr_info, iStepOffset);

	#ifdef PERFORMANCE_METRICS
		cudaLapTime(clr_info, clr_info->t_gpu_advanceNeurons);
	#endif
}

/*
 * Process outgoing spiking data between clusters.
 *
 * @param  clr_info  ClusterInfo to refer.
 */
void GPUSpikingCluster::processInterClustesOutgoingSpikes(ClusterInfo *clr_info)
{
	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

	// wait until all CUDA related tasks complete
	checkCudaErrors( cudaDeviceSynchronize() );

	#ifdef PERFORMANCE_METRICS
		cudaStartTimer(clr_info);
	#endif

	// process inter clusters outgoing spikes
	dynamic_cast<AllSpikingSynapses*>(m_synapses)->processInterClustesOutgoingSpikes(m_allSynapsesDevice);

	#ifdef PERFORMANCE_METRICS
		cudaLapTime(clr_info, clr_info->t_gpu_processInterClustesOutgoingSpikes);
	#endif
}

/*
 * Process incoming spiking data between clusters.
 *
 * @param  clr_info  ClusterInfo to refer.
 */
void GPUSpikingCluster::processInterClustesIncomingSpikes(ClusterInfo *clr_info)
{
	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

	// wait until all CUDA related tasks complete
	checkCudaErrors( cudaDeviceSynchronize() );

	#ifdef PERFORMANCE_METRICS
		cudaStartTimer(clr_info);
	#endif

	// process inter clusters incoming spikes
	dynamic_cast<AllSpikingSynapses*>(m_synapses)->processInterClustesIncomingSpikes(m_allSynapsesDevice);

	#ifdef PERFORMANCE_METRICS
		cudaLapTime(clr_info, clr_info->t_gpu_processInterClustesIncomingSpikes);
	#endif
}

/*
 * Advances synapses network state of the cluster one simulation step.
 *
 * @param sim_info   parameters defining the simulation to be run with
 *                   the given collection of neurons.
 * @param clr_info   ClusterInfo to refer.
 * @param iStepOffset  Offset from the current simulation step.
 */
void GPUSpikingCluster::advanceSynapses(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset)
{
	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) ); // Set device ID

	#ifdef PERFORMANCE_METRICS
		cudaStartTimer(clr_info);
	#endif

	m_synapses->advanceSynapses(m_allSynapsesDevice, m_allNeuronsDevice, m_synapseIndexMapDevice, sim_info, clr_info, iStepOffset);

	#ifdef PERFORMANCE_METRICS
		cudaLapTime(clr_info, clr_info->t_gpu_advanceSynapses);
		cudaStartTimer(clr_info);
	#endif

	// calculate summation point
    calcSummationMap(sim_info, clr_info);

	#ifdef PERFORMANCE_METRICS
		cudaLapTime(clr_info, clr_info->t_gpu_calcSummation);
	#endif

	// wait until all CUDA related tasks complete
	checkCudaErrors( cudaDeviceSynchronize() );
}

/*
 * Advances synapses spike event queue state of the cluster one simulation step.
 *
 * @param sim_info - parameters defining the simulation to be run with
 *                   the given collection of neurons.
 * @param clr_info - parameters defining the simulation to be run with
 *                   the given collection of neurons.
 * @param iStep    - simulation steps to advance.
 */
void GPUSpikingCluster::advanceSpikeQueue(const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStep)
{
	checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

	(dynamic_cast<AllSpikingSynapses*>(m_synapses))->advanceSpikeQueue(m_allSynapsesDevice, iStep);

	if (sim_info->pInput != NULL) 
		sim_info->pInput->advanceSInputState(clr_info, iStep); // advance input stimulus state
	
	// wait until all CUDA related tasks complete
	checkCudaErrors( cudaDeviceSynchronize() );
}

/*
 * Add psr of all incoming synapses to summation points.
 * (sequential addtion base summation)
 *
 * @param[in] sim_info                   Pointer to the simulation information.
 * @param[in] clr_info                   Pointer to the cluster information.
 */
void GPUSpikingCluster::calcSummationMap(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
	if (m_synapseIndexMap == NULL) return;
	BGSIZE numTotalSynapses = m_synapseIndexMap->num_incoming_synapses;
	if (numTotalSynapses == 0) return;

	// call sequential addtion base summation kernel
	calcSummationMapDevice <<< clr_info->neuronBlocksPerGrid, clr_info->threadsPerBlock >>> (clr_info->totalClusterNeurons, m_allNeuronsDevice, m_synapseIndexMapDevice, m_allSynapsesDevice);
}

/*
 *  Allocate device memory for synapse index map.
 *  @param  count	The number of neurons.
 */
void GPUSpikingCluster::allocSynapseImap( int count )
{
	SynapseIndexMap synapseIndexMap;

	checkCudaErrors(cudaMalloc((void **) &synapseIndexMap.outgoingSynapseBegin, count * sizeof(BGSIZE)));
	checkCudaErrors(cudaMalloc((void **) &synapseIndexMap.outgoingSynapseCount, count * sizeof(BGSIZE)));
	checkCudaErrors(cudaMemset(synapseIndexMap.outgoingSynapseBegin, 0, count * sizeof(BGSIZE)));
	checkCudaErrors(cudaMemset(synapseIndexMap.outgoingSynapseCount, 0, count * sizeof(BGSIZE)));

	checkCudaErrors(cudaMalloc((void **) &synapseIndexMap.incomingSynapseBegin, count * sizeof(BGSIZE)));
	checkCudaErrors(cudaMalloc((void **) &synapseIndexMap.incomingSynapseCount, count * sizeof(BGSIZE)));
	checkCudaErrors(cudaMemset(synapseIndexMap.incomingSynapseBegin, 0, count * sizeof(BGSIZE)));
	checkCudaErrors(cudaMemset(synapseIndexMap.incomingSynapseCount, 0, count * sizeof(BGSIZE)));

	checkCudaErrors(cudaMalloc((void **) &m_synapseIndexMapDevice, sizeof(SynapseIndexMap)));
	checkCudaErrors(cudaMemcpy(m_synapseIndexMapDevice, &synapseIndexMap, sizeof(SynapseIndexMap), cudaMemcpyHostToDevice));
}

/*
 *  Deallocate device memory for synapse inverse map.
 */
void GPUSpikingCluster::deleteSynapseImap(  )
{
	SynapseIndexMap synapseIndexMap;

	checkCudaErrors(cudaMemcpy(&synapseIndexMap, m_synapseIndexMapDevice, sizeof(SynapseIndexMap), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(synapseIndexMap.outgoingSynapseBegin));
	checkCudaErrors(cudaFree(synapseIndexMap.outgoingSynapseCount));
	checkCudaErrors(cudaFree(synapseIndexMap.outgoingSynapseIndexMap));

	checkCudaErrors(cudaFree(synapseIndexMap.incomingSynapseBegin));
	checkCudaErrors(cudaFree(synapseIndexMap.incomingSynapseCount));
	checkCudaErrors(cudaFree(synapseIndexMap.incomingSynapseIndexMap));

	checkCudaErrors(cudaFree(m_synapseIndexMapDevice));
}

/* 
 *  Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
 *
 *  @param  clr_info    ClusterInfo to refer from.
 */
void GPUSpikingCluster::copySynapseIndexMapHostToDevice(const ClusterInfo *clr_info)
{
	checkCudaErrors(cudaSetDevice(clr_info->deviceId));

	SynapseIndexMap *synapseIndexMapHost = m_synapseIndexMap;
	SynapseIndexMap *synapseIndexMapDevice = m_synapseIndexMapDevice;
	int total_synapse_counts = dynamic_cast<AllSynapses*>(m_synapses)->total_synapse_counts;
	int neuron_count = clr_info->totalClusterNeurons;

	if (synapseIndexMapHost == NULL || total_synapse_counts == 0) return;

	SynapseIndexMap synapseIndexMap;

	checkCudaErrors(cudaMemcpy(&synapseIndexMap, synapseIndexMapDevice, sizeof(SynapseIndexMap), cudaMemcpyDeviceToHost));

	// outgoing synaps index map
	checkCudaErrors(cudaMemcpy(synapseIndexMap.outgoingSynapseBegin, synapseIndexMapHost->outgoingSynapseBegin, neuron_count * sizeof(BGSIZE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(synapseIndexMap.outgoingSynapseCount, synapseIndexMapHost->outgoingSynapseCount, neuron_count * sizeof(BGSIZE), cudaMemcpyHostToDevice));
	
	// the number of synapses may change, so we reallocate the memory
	if (synapseIndexMap.outgoingSynapseIndexMap != NULL) 
		checkCudaErrors(cudaFree(synapseIndexMap.outgoingSynapseIndexMap));
	
	checkCudaErrors(cudaMalloc((void **) &synapseIndexMap.outgoingSynapseIndexMap, total_synapse_counts * sizeof(OUTGOING_SYNAPSE_INDEX_TYPE)));
	checkCudaErrors(cudaMemcpy(synapseIndexMap.outgoingSynapseIndexMap, synapseIndexMapHost->outgoingSynapseIndexMap, total_synapse_counts * sizeof(OUTGOING_SYNAPSE_INDEX_TYPE), cudaMemcpyHostToDevice));

	// incomming synapse index map
	checkCudaErrors(cudaMemcpy(synapseIndexMap.incomingSynapseBegin, synapseIndexMapHost->incomingSynapseBegin, neuron_count * sizeof(BGSIZE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(synapseIndexMap.incomingSynapseCount, synapseIndexMapHost->incomingSynapseCount, neuron_count * sizeof(BGSIZE), cudaMemcpyHostToDevice));
	
	// the number of synapses may change, so we reallocate the memory
	if (synapseIndexMap.incomingSynapseIndexMap != NULL) 
		checkCudaErrors( cudaFree( synapseIndexMap.incomingSynapseIndexMap ) );
	
	checkCudaErrors(cudaMalloc((void **) &synapseIndexMap.incomingSynapseIndexMap, total_synapse_counts * sizeof(BGSIZE)));
	checkCudaErrors(cudaMemcpy(synapseIndexMap.incomingSynapseIndexMap, synapseIndexMapHost->incomingSynapseIndexMap, total_synapse_counts * sizeof(BGSIZE), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(synapseIndexMapDevice, &synapseIndexMap, sizeof(SynapseIndexMap), cudaMemcpyHostToDevice));

	DEBUG({reportGPUMemoryUsage(clr_info);})
}

/**
 * Calculate the sum of synaptic input to each neuron.
 *
 * Calculate the sum of synaptic input to each neuron. One thread
 * corresponds to one neuron. Iterates sequentially through the
 * forward synapse index map (synapseIndexMapDevice) to access only
 * existing synapses. Using this structure eliminates the need to skip
 * synapses that have undergone lazy deletion from the main
 * (allSynapsesDevice) synapse structure. The forward map is
 * re-computed during each network restructure (once per epoch) to
 * ensure that all synapse pointers for a neuron are stored
 * contiguously.
 * 
 * @param[in] totalNeurons           Number of neurons in the entire simulation.
 * @param[in,out] allNeuronsDevice   Pointer to Neuron structures in device memory.
 * @param[in] synapseIndexMapDevice  Pointer to forward map structures in device memory.
 * @param[in] allSynapsesDevice      Pointer to Synapse structures in device memory.
 */
__global__ void calcSummationMapDevice(const int totalNeurons, 
				       AllSpikingNeuronsDeviceProperties* __restrict__ allNeuronsDevice, 
				       const SynapseIndexMap* __restrict__ synapseIndexMapDevice, 
				       const AllSpikingSynapsesDeviceProperties* __restrict__ allSynapsesDevice)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= totalNeurons )
		return;

	const BGSIZE synCount = synapseIndexMapDevice->incomingSynapseCount[idx];

	if (synCount != 0) {
		const int beginIndex = synapseIndexMapDevice->incomingSynapseBegin[idx];
		const BGSIZE* activeMap_begin = 
			&(synapseIndexMapDevice->incomingSynapseIndexMap[beginIndex]);
		BGFLOAT sum = 0.0;
		BGSIZE synIndex;

		for (BGSIZE i = 0; i < synCount; i++) {
			synIndex = activeMap_begin[i];
			sum += allSynapsesDevice->psr[synIndex];
		}
		
		allNeuronsDevice->summation_map[idx] = sum; // Store summed PSR into this neuron's summation point
	}
}
