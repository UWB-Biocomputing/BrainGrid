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

__global__ void setSynapseSummationPointDevice(int num_neurons, AllSpikingNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, int max_synapses, int width);

//! Calculate summation point.
__global__ void calcSummationMapDevice( int totalNeurons, SynapseIndexMap* synapseIndexMapDevice, AllSpikingSynapses* allSynapsesDevice );

//! Update the network.
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, void (*fpCreateSynapse)(AllSpikingSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType) );

//! Add a synapse to the network.
__device__ void addSynapse( AllSpikingSynapses* allSynapsesDevice, synapseType type, const int src_neuron, const int dest_neuron, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons, void (*fpCreateSynapse)(AllSpikingSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType) );

//! Remove a synapse from the network.
__device__ void eraseSynapse( AllSpikingSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int maxSynapses );

//! Get the type of synapse.
__device__ synapseType synType( AllSpikingNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron );

//! Get the type of synapse (excitatory or inhibitory)
__device__ int synSign( synapseType t );

#ifdef PERFORMANCE_METRICS
float g_time;
cudaEvent_t start, stop;
#endif // PERFORMANCE_METRICS

// ----------------------------------------------------------------------------

GPUSpikingModel::GPUSpikingModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) : 	
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

	// allocate synapse inverse map in device memory
	allocSynapseImap( neuron_count );
}

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

    // allocates memories on CUDA device
    allocDeviceStruct((void **)&m_allNeuronsDevice, (void **)&m_allSynapsesDevice, sim_info);

    // set device summation points
    int neuron_count = sim_info->totalNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
    setSynapseSummationPointDevice <<< blocksPerGrid, threadsPerBlock >>> (neuron_count, m_allNeuronsDevice, m_allSynapsesDevice, sim_info->maxSynapsesPerNeuron, sim_info->width);
}

/** 
*  Begin terminating the simulator.
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

/**
 *  Loads the simulation based on istream input.
 *  @param  input   istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void GPUSpikingModel::loadMemory(istream& input, const SimulationInfo *sim_info)
{
    Model::loadMemory(input, sim_info);

    // copy inverse map to the device memory
    copySynapseIndexMapHostToDevice(*m_synapseIndexMap, sim_info->totalNeurons, m_synapses->total_synapse_counts);

    // Reinitialize device struct - Copy host neuron and synapse arrays into GPU device
    m_neurons->copyNeuronHostToDevice( m_allNeuronsDevice, sim_info );
    m_synapses->copySynapseHostToDevice( m_allSynapsesDevice, sim_info );

    // set summation points
    int neuron_count = sim_info->totalNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
    setSynapseSummationPointDevice <<< blocksPerGrid, threadsPerBlock >>> (neuron_count, m_allNeuronsDevice, m_allSynapsesDevice, sim_info->maxSynapsesPerNeuron, sim_info->width);
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

void GPUSpikingModel::calcSummationMap(const SimulationInfo *sim_info)
{
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;

    calcSummationMapDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, synapseIndexMapDevice, m_allSynapsesDevice );
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
*  Update the weight of the Synapses in the simulation.
*  Note: Platform Dependent.
*  @param  num_neurons number of neurons to update.
*  @param  neurons the Neuron list to search from.
*  @param  synapses    the Synapse list to search from.
*  @param  sim_info    SimulationInfo to refer from.
*/
void GPUSpikingModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
{
        // For now, we just set the weights to equal the areas. We will later
        // scale it and set its sign (when we index and get its sign).
        (*m_conns->W) = (*m_conns->area);

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
                        W_h[i * sim_info->totalNeurons + j] = (*m_conns->W)(i, j);

        HANDLE_ERROR( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

        unsigned long long fpCreateSynapse_h;
        synapses.getFpCreateSynapse(fpCreateSynapse_h); 

        blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
        updateNetworkDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, width, deltaT, W_d, sim_info->maxSynapsesPerNeuron, m_allNeuronsDevice, m_allSynapsesDevice, (void (*)(AllSpikingSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType))fpCreateSynapse_h );

        // free memories
        HANDLE_ERROR( cudaFree( W_d ) );
        delete[] W_h;

        // copy device synapse count to host memory
        synapses.copyDeviceSynapseCountsToHost(m_allSynapsesDevice, sim_info);
        // copy device synapse summation coordinate to host memory
        synapses.copyDeviceSynapseSumCoordToHost(m_allSynapsesDevice, sim_info);
        // create synapse inverse map
        createSynapseImap( synapses, sim_info );
        // copy inverse map to the device memory
        copySynapseIndexMapHostToDevice(*m_synapseIndexMap, sim_info->totalNeurons, synapses.total_synapse_counts);
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
    neurons.copyNeuronDeviceSpikeCountsToHost(m_allNeuronsDevice, sim_info);
    neurons.copyNeuronDeviceSpikeHistoryToHost(m_allNeuronsDevice, sim_info);

    Model::updateHistory(currentStep, epochDuration, sim_info, simRecorder);

    // clear spike count
    neurons.clearNeuronSpikeCounts(m_allNeuronsDevice, sim_info);
}

/* ------------------*\
|* # Global Functions
\* ------------------*/

/**
 * Set the summation points in device memory
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
            int dest_neuron = allSynapsesDevice->summationCoord[max_synapses * src_neuron + syn_index].x
                + allSynapsesDevice->summationCoord[max_synapses * src_neuron + syn_index].y * width;
            allSynapsesDevice->summationPoint[max_synapses * src_neuron + syn_index] = &( allNeuronsDevice->summation_map[dest_neuron] );
            n_inUse++;
        }
    }
}

/** 
* @param[in] totalNeurons       Number of neurons.
* @param[in] synapseIndexMap    Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void calcSummationMapDevice( int totalNeurons, SynapseIndexMap* synapseIndexMapDevice, AllSpikingSynapses* allSynapsesDevice ) {
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
* @param[in] num_neurons        Number of neurons.
* @param[in] width              Width of neuron map (assumes square).
* @param[in] deltaT             The time step size.
* @param[in] W_d                Array of synapse weight.
* @param[in] maxSynapses        Maximum number of synapses per neuron.
* @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
* @param[in] allSynapsesDevice         Pointer to the Synapse structures in device memory.
*/
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, void (*fpCreateSynapse)(AllSpikingSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType) )
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

            addSynapse(allSynapsesDevice, type, src_neuron, dest_neuron, xa, ya, xb, yb, sum_point, deltaT, W_d, num_neurons, fpCreateSynapse);

        }
    }
}

/** 
* Adds a synapse to the network.  Requires the locations of the source and
* destination neurons.
* @param allSynapsesDevice      Pointer to the Synapse structures in device memory.
* @param type                   Type of the Synapse to create.
* @param src_neuron             Index of the source neuron.
* @param dest_neuron            Index of the destination neuron.
* @param source_x               X location of source.
* @param source_y               Y location of source.
* @param dest_x                 X location of destination.
* @param dest_y                 Y location of destination.
* @param sum_point              Pointer to the summation point.
* @param deltaT                 The time step size.
* @param W_d                    Array of synapse weight.
* @param num_neurons            The number of neurons.
*/
__device__ void addSynapse(AllSpikingSynapses* allSynapsesDevice, synapseType type, const int src_neuron, const int dest_neuron, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons, void (*fpCreateSynapse)(AllSpikingSynapses*, const int, const int, int, int, int, int, BGFLOAT*, const BGFLOAT, synapseType))
{
    if (allSynapsesDevice->synapse_counts[src_neuron] >= allSynapsesDevice->maxSynapsesPerNeuron) {
        return; // TODO: ERROR!
    }

    // add it to the list
    size_t synapse_index;
    size_t max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
    uint32_t iSync = max_synapses * src_neuron;
    for (synapse_index = 0; synapse_index < max_synapses; synapse_index++) {
        if (!allSynapsesDevice->in_use[iSync + synapse_index]) {
            break;
        }
    }

    allSynapsesDevice->synapse_counts[src_neuron]++;

    // create a synapse
    fpCreateSynapse(allSynapsesDevice, src_neuron, synapse_index, source_x, source_y, dest_x, dest_y, sum_point, deltaT, type );
    allSynapsesDevice->W[iSync + synapse_index] = W_d[src_neuron * num_neurons + dest_neuron] * synSign(type) * SYNAPSE_STRENGTH_ADJUSTMENT;
}

/** 
* Remove a synapse from the network.
* @param[in] allSynapsesDevice         Pointer to the Synapse structures in device memory.
* @param neuron_index   Index of a neuron.
* @param synapse_index  Index of a synapse.
* @param[in] maxSynapses        Maximum number of synapses per neuron.
*/
__device__ void eraseSynapse( AllSpikingSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int maxSynapses )
{
    uint32_t iSync = maxSynapses * neuron_index + synapse_index;
    allSynapsesDevice->synapse_counts[neuron_index]--;
    allSynapsesDevice->in_use[iSync] = false;
    allSynapsesDevice->summationPoint[iSync] = NULL;
}

/** 
* Returns the type of synapse at the given coordinates
* @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
* @param src_neuron             Index of the source neuron.
* @param dest_neuron            Index of the destination neuron.
*/
__device__ synapseType synType( AllSpikingNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron )
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
* @param[in] t  synapseType I to I, I to E, E to I, or E to E
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

