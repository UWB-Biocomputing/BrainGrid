/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - **\ 
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

#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif // PERFORMANCE_METRICS

__constant__ int d_debug_mask[1];

// ----------------------------------------------------------------------------

GPUSpikingModel::GPUSpikingModel(Connections *conns, IAllNeurons *neurons, IAllSynapses *synapses, Layout *layout) : 	
	Model::Model(conns, neurons, synapses, layout),
	synapseIndexMapDevice(NULL),
	randNoise_d(NULL),
	m_allNeuronsDevice(NULL),
	m_allSynapsesDevice(NULL)
{
}

GPUSpikingModel::~GPUSpikingModel() 
{
	//Let Model base class handle de-allocation
}

/*
 * Allocates and initializes memories on CUDA device.
 *
 * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
 * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
 * @param[in]  sim_info			Pointer to the simulation information.
 */
void GPUSpikingModel::allocDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info)
{
	// Allocate Neurons and Synapses strucs on GPU device memory
	m_neurons->allocNeuronDeviceStruct( allNeuronsDevice, sim_info );
	m_synapses->allocSynapseDeviceStruct( allSynapsesDevice, sim_info );

	// Allocate memory for random noise array
	int neuron_count = sim_info->totalNeurons;
	BGSIZE randNoise_d_size = neuron_count * sizeof (float);	// size of random noise array
	HANDLE_ERROR( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );

	// Copy host neuron and synapse arrays into GPU device
	m_neurons->copyNeuronHostToDevice( *allNeuronsDevice, sim_info );
	m_synapses->copySynapseHostToDevice( *allSynapsesDevice, sim_info );

	// allocate synapse inverse map in device memory
	allocSynapseImap( neuron_count );
}

/*
 * Copies device memories to host memories and deallocaes them.
 *
 * @param[out] allNeuronsDevice          Memory loation of the pointer to the neurons list on device memory.
 * @param[out] allSynapsesDevice         Memory loation of the pointer to the synapses list on device memory.
 * @param[in]  sim_info                  Pointer to the simulation information.
 */
void GPUSpikingModel::deleteDeviceStruct(void** allNeuronsDevice, void** allSynapsesDevice, SimulationInfo *sim_info)
{
    // copy device synapse and neuron structs to host memory
    m_neurons->copyNeuronDeviceToHost( *allNeuronsDevice, sim_info );

    // Deallocate device memory
    m_neurons->deleteNeuronDeviceStruct( *allNeuronsDevice, sim_info );

    // copy device synapse and neuron structs to host memory
    m_synapses->copySynapseDeviceToHost( *allSynapsesDevice, sim_info );

    // Deallocate device memory
    m_synapses->deleteSynapseDeviceStruct( *allSynapsesDevice );

    deleteSynapseImap();

    HANDLE_ERROR( cudaFree( randNoise_d ) );
}

/*
 *  Sets up the Simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void GPUSpikingModel::setupSim(SimulationInfo *sim_info)
{
    // Set device ID
    HANDLE_ERROR( cudaSetDevice( g_deviceId ) );

    // Set DEBUG flag
    HANDLE_ERROR( cudaMemcpyToSymbol (d_debug_mask, &g_debug_mask, sizeof(int) ) );

    Model::setupSim(sim_info);

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

    // allocates memories on CUDA device
    allocDeviceStruct((void **)&m_allNeuronsDevice, (void **)&m_allSynapsesDevice, sim_info);

    // set device summation points
    int neuron_count = sim_info->totalNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
    setSynapseSummationPointDevice <<< blocksPerGrid, threadsPerBlock >>> (neuron_count, m_allNeuronsDevice, m_allSynapsesDevice, sim_info->maxSynapsesPerNeuron, sim_info->width);

    // copy inverse map to the device memory
    copySynapseIndexMapHostToDevice(*m_synapseIndexMap, sim_info->totalNeurons);

    // set some parameters used for advanceNeuronsDevice
    m_neurons->setAdvanceNeuronsDeviceParams(*m_synapses);

    // set some parameters used for advanceSynapsesDevice
    m_synapses->setAdvanceSynapsesDeviceParams();
}

/* 
 *  Begin terminating the simulator.
 *
 *  @param  sim_info    SimulationInfo to refer.
 */
void GPUSpikingModel::cleanupSim(SimulationInfo *sim_info)
{
    // deallocates memories on CUDA device
    deleteDeviceStruct((void**)&m_allNeuronsDevice, (void**)&m_allSynapsesDevice, sim_info);

#ifdef PERFORMANCE_METRICS
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
#endif // PERFORMANCE_METRICS
}

/*
 *  Loads the simulation based on istream input.
 *
 *  @param  input   istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void GPUSpikingModel::deserialize(istream& input, const SimulationInfo *sim_info)
{
    Model::deserialize(input, sim_info);

    // copy inverse map to the device memory
    copySynapseIndexMapHostToDevice(*m_synapseIndexMap, sim_info->totalNeurons);

    // Reinitialize device struct - Copy host neuron and synapse arrays into GPU device
    m_neurons->copyNeuronHostToDevice( m_allNeuronsDevice, sim_info );
    m_synapses->copySynapseHostToDevice( m_allSynapsesDevice, sim_info );

    // set summation points
    int neuron_count = sim_info->totalNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
    setSynapseSummationPointDevice <<< blocksPerGrid, threadsPerBlock >>> (neuron_count, m_allNeuronsDevice, m_allSynapsesDevice, sim_info->maxSynapsesPerNeuron, sim_info->width);
}

/* 
 *  Advance everything in the model one time step. In this case, that
 *  means calling all of the kernels that do the "micro step" updating
 *  (i.e., NOT the stuff associated with growth).
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void GPUSpikingModel::advance(const SimulationInfo *sim_info)
{
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
	m_neurons->advanceNeurons(*m_synapses, m_allNeuronsDevice, m_allSynapsesDevice, sim_info, randNoise_d, synapseIndexMapDevice);

#ifdef PERFORMANCE_METRICS
	lapTime(t_gpu_advanceNeurons);
	startTimer();
#endif // PERFORMANCE_METRICS

	// Advance synapses ------------->
	m_synapses->advanceSynapses(m_allSynapsesDevice, m_allNeuronsDevice, synapseIndexMapDevice, sim_info);

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

/*
 * Add psr of all incoming synapses to summation points.
 *
 * @param[in] sim_info                   Pointer to the simulation information.
 */
void GPUSpikingModel::calcSummationMap(const SimulationInfo *sim_info)
{
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;

    calcSummationMapDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, synapseIndexMapDevice, m_allSynapsesDevice );
}

/* 
 *  Update the connection of all the Neurons and Synapses of the simulation.
 *
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void GPUSpikingModel::updateConnections(const SimulationInfo *sim_info)
{
        dynamic_cast<AllSpikingNeurons*>(m_neurons)->copyNeuronDeviceSpikeCountsToHost(m_allNeuronsDevice, sim_info);
        dynamic_cast<AllSpikingNeurons*>(m_neurons)->copyNeuronDeviceSpikeHistoryToHost(m_allNeuronsDevice, sim_info);

        // Update Connections data
        if (m_conns->updateConnections(*m_neurons, sim_info, m_layout)) {
	    m_conns->updateSynapsesWeights(sim_info->totalNeurons, *m_neurons, *m_synapses, sim_info, m_allNeuronsDevice, m_allSynapsesDevice, m_layout);
            // create synapse inverse map
            m_synapses->createSynapseImap(m_synapseIndexMap, sim_info);
            // copy inverse map to the device memory
            copySynapseIndexMapHostToDevice(*m_synapseIndexMap, sim_info->totalNeurons);
        }
}

/*
 *  Update the Neuron's history.
 *
 *  @param  sim_info    SimulationInfo to refer from.
 */
void GPUSpikingModel::updateHistory(const SimulationInfo *sim_info)
{
    Model::updateHistory(sim_info);

    // clear spike count
    dynamic_cast<AllSpikingNeurons*>(m_neurons)->clearNeuronSpikeCounts(m_allNeuronsDevice, sim_info);
}

/* ------------------*\
|* # Helper Functions
\* ------------------*/

/*
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

/*
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

/* 
 *  Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
 *
 *  @param  synapseIndexMapHost		Reference to the SynapseIndexMap in host memory.
 *  @param  neuron_count		The number of neurons.
 */
void GPUSpikingModel::copySynapseIndexMapHostToDevice(SynapseIndexMap &synapseIndexMapHost, int neuron_count)
{
        int total_synapse_counts = dynamic_cast<AllSynapses*>(m_synapses)->total_synapse_counts;

	if (total_synapse_counts == 0)
		return;

	SynapseIndexMap synapseIndexMap;

	HANDLE_ERROR( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.incomingSynapse_begin, synapseIndexMapHost.incomingSynapse_begin, neuron_count * sizeof( int ), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.synapseCount, synapseIndexMapHost.synapseCount, neuron_count * sizeof( int ), cudaMemcpyHostToDevice ) );
	// the number of synapses may change, so we reallocate the memory
	if (synapseIndexMap.inverseIndex != NULL) {
		HANDLE_ERROR( cudaFree( synapseIndexMap.inverseIndex ) );
	}
	HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.inverseIndex, total_synapse_counts * sizeof( BGSIZE ) ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.inverseIndex, synapseIndexMapHost.inverseIndex, total_synapse_counts * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );

	if (synapseIndexMap.activeSynapseIndex != NULL) {
		HANDLE_ERROR( cudaFree( synapseIndexMap.activeSynapseIndex ) );
	}
	HANDLE_ERROR( cudaMalloc( ( void ** ) &synapseIndexMap.activeSynapseIndex, total_synapse_counts * sizeof( BGSIZE ) ) );
	HANDLE_ERROR( cudaMemcpy ( synapseIndexMap.activeSynapseIndex, synapseIndexMapHost.activeSynapseIndex, total_synapse_counts * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );

	HANDLE_ERROR( cudaMemcpy ( synapseIndexMapDevice, &synapseIndexMap, sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

/*
 * Set the summation points in device memory
 *
 * @param[in] num_neurons        Number of neurons.
 * @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
 * @param[in] max_synapses       Maximum number of synapses per neuron.
 * @param[in] width              Width of neuron map (assumes square).
 */
__global__ void setSynapseSummationPointDevice(int num_neurons, AllSpikingNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, int max_synapses, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_neurons )
        return;

    int src_neuron = idx;
    int n_inUse = 0;
    for (int syn_index = 0; n_inUse < allSynapsesDevice->synapse_counts[src_neuron]; syn_index++) {
        if (allSynapsesDevice->in_use[max_synapses * src_neuron + syn_index] == true) {
            int dest_neuron = allSynapsesDevice->destNeuronIndex[max_synapses * src_neuron + syn_index];
            allSynapsesDevice->summationPoint[max_synapses * src_neuron + syn_index] = &( allNeuronsDevice->summation_map[dest_neuron] );
            n_inUse++;
        }
    }
}

/* 
 * @param[in] totalNeurons       Number of neurons.
 * @param[in] synapseIndexMap    Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 * @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
 */
__global__ void calcSummationMapDevice( int totalNeurons, SynapseIndexMap* synapseIndexMapDevice, AllSpikingSynapses* allSynapsesDevice ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalNeurons )
                return;

        BGSIZE iCount = synapseIndexMapDevice->synapseCount[idx];
        if (iCount != 0) {
                int beginIndex = synapseIndexMapDevice->incomingSynapse_begin[idx];
                BGSIZE* inverseMap_begin = &( synapseIndexMapDevice->inverseIndex[beginIndex] );
                BGFLOAT sum = 0.0;
                BGSIZE syn_i = inverseMap_begin[0];
                BGFLOAT &summationPoint = *( allSynapsesDevice->summationPoint[syn_i] );
                for ( BGSIZE i = 0; i < iCount; i++ ) {
                        syn_i = inverseMap_begin[i];
                        sum += allSynapsesDevice->psr[syn_i];
                }
                summationPoint = sum;
        }
}

