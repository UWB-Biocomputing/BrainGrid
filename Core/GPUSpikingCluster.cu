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
#include "ISInput.h"

// ----------------------------------------------------------------------------

GPUSpikingCluster::GPUSpikingCluster(IAllNeurons *neurons, IAllSynapses *synapses) : 	
  Cluster::Cluster(neurons, synapses),
  m_synapseIndexMapDevice(NULL),
  randNoise_d(NULL),
  m_allNeuronsDevice(NULL),
  m_allSynapsesDevice(NULL)
{
}

GPUSpikingCluster::~GPUSpikingCluster() 
{
  // Let Cluster base class handle de-allocation
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
  // Allocate Neurons and Synapses strucs on GPU device memory
  m_neurons->allocNeuronDeviceStruct( allNeuronsDevice, sim_info, clr_info );
  m_synapses->allocSynapseDeviceStruct( allSynapsesDevice, sim_info, clr_info );

  // Allocate memory for random noise array
  int neuron_count = clr_info->totalClusterNeurons;
  // neuron_count must be a multiple of 100
  neuron_count = ((( neuron_count - 1 ) / 100 ) + 1 ) * 100;
  BGSIZE randNoise_d_size = neuron_count * sizeof (float);	// size of random noise array
  checkCudaErrors( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );

  // Copy host neuron and synapse arrays into GPU device
  m_neurons->copyNeuronHostToDevice( *allNeuronsDevice, sim_info, clr_info );
  m_synapses->copySynapseHostToDevice( *allSynapsesDevice, sim_info, clr_info );

  // allocate synapse index map in device memory
  allocSynapseImap( neuron_count );
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
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

  Cluster::setupCluster(sim_info, layout, clr_info);

  //initialize Mersenne Twister
  //assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
  int rng_blocks = 25; //# of blocks the kernel will use
  int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
  int neuron_count = clr_info->totalClusterNeurons;
  // neuron_count must be a multiple of 100
  neuron_count = ((( neuron_count - 1 ) / 100 ) + 1 ) * 100;
  int rng_mt_rng_count = neuron_count/rng_nPerRng; //# of threads to generate for neuron_count rand #s
  int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed
  initMTGPU(clr_info->seed, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

#ifdef PERFORMANCE_METRICS
  cudaEventCreate( &clr_info->start );
  cudaEventCreate( &clr_info->stop );

  clr_info->t_gpu_rndGeneration = 0.0;
  clr_info->t_gpu_advanceNeurons = 0.0;
  clr_info->t_gpu_advanceSynapses = 0.0;
  clr_info->t_gpu_calcSummation = 0.0;
#endif // PERFORMANCE_METRICS

  // allocates memories on CUDA device
  allocDeviceStruct((void **)&m_allNeuronsDevice, (void **)&m_allSynapsesDevice, sim_info, clr_info);

  // set some parameters used for advanceNeuronsDevice
  m_neurons->setAdvanceNeuronsDeviceParams(*m_synapses);

  // set some parameters used for advanceSynapsesDevice
  m_synapses->setAdvanceSynapsesDeviceParams();
}

/* 
 *  Begin terminating the simulator.
 *
 *  @param  sim_info    SimulationInfo to refer.
 *  @param  clr_info    ClusterInfo to refer.
 */
void GPUSpikingCluster::cleanupCluster(SimulationInfo *sim_info, ClusterInfo *clr_info)
{
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

  // deallocates memories on CUDA device
  deleteDeviceStruct((void**)&m_allNeuronsDevice, (void**)&m_allSynapsesDevice, sim_info, clr_info);

#ifdef PERFORMANCE_METRICS
  cudaEventDestroy( clr_info->start );
  cudaEventDestroy( clr_info->stop );
#endif // PERFORMANCE_METRICS

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
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

#ifdef PERFORMANCE_METRICS
  // Reset CUDA timer to start measurement of GPU operation
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  normalMTGPU(randNoise_d);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_rndGeneration);
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // Advance neurons ------------->
  m_neurons->advanceNeurons(*m_synapses, m_allNeuronsDevice, m_allSynapsesDevice, sim_info, randNoise_d, m_synapseIndexMapDevice, clr_info, iStepOffset);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_advanceNeurons);
#endif // PERFORMANCE_METRICS

}

/*
 * Process outgoing spiking data between clusters.
 *
 * @param  clr_info  ClusterInfo to refer.
 */
void GPUSpikingCluster::processInterClustesOutgoingSpikes(ClusterInfo *clr_info)
{
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

  // wait until all CUDA related tasks complete
  checkCudaErrors( cudaDeviceSynchronize() );

  // process inter clusters outgoing spikes
  dynamic_cast<AllSpikingSynapses*>(m_synapses)->processInterClustesOutgoingSpikes(m_allSynapsesDevice);
}

/*
 * Process incoming spiking data between clusters.
 *
 * @param  clr_info  ClusterInfo to refer.
 */
void GPUSpikingCluster::processInterClustesIncomingSpikes(ClusterInfo *clr_info)
{
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

  // wait until all CUDA related tasks complete
  checkCudaErrors( cudaDeviceSynchronize() );

  // process inter clusters incoming spikes
  dynamic_cast<AllSpikingSynapses*>(m_synapses)->processInterClustesIncomingSpikes(m_allSynapsesDevice);
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
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

#ifdef PERFORMANCE_METRICS
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // Advance synapses ------------->
  m_synapses->advanceSynapses(m_allSynapsesDevice, m_allNeuronsDevice, m_synapseIndexMapDevice, sim_info, clr_info, iStepOffset);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_advanceSynapses);
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // calculate summation point
  calcSummationMap(sim_info, clr_info);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_calcSummation);
#endif // PERFORMANCE_METRICS

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
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

  (dynamic_cast<AllSpikingSynapses*>(m_synapses))->advanceSpikeQueue(m_allSynapsesDevice, iStep);

  if (sim_info->pInput != NULL) {
      // advance input stimulus state
      sim_info->pInput->advanceSInputState(clr_info, iStep);
  }

  // wait until all CUDA related tasks complete
  checkCudaErrors( cudaDeviceSynchronize() );
}

/*
 * Add psr of all incoming synapses to summation points.
 *
 * @param[in] sim_info                   Pointer to the simulation information.
 * @param[in] clr_info                   Pointer to the cluster information.
 */
void GPUSpikingCluster::calcSummationMap(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
  // CUDA parameters
  const int threadsPerBlock = 256;
  int blocksPerGrid = ( clr_info->totalClusterNeurons + threadsPerBlock - 1 ) / threadsPerBlock;

  calcSummationMapDevice <<< blocksPerGrid, threadsPerBlock >>> ( clr_info->totalClusterNeurons, m_allNeuronsDevice, m_synapseIndexMapDevice, m_allSynapsesDevice );
}

/* ------------------*\
   |* # Helper Functions
   \* ------------------*/

/*
 *  Allocate device memory for synapse index map.
 *  @param  count	The number of neurons.
 */
void GPUSpikingCluster::allocSynapseImap( int count )
{
  SynapseIndexMap synapseIndexMap;

  checkCudaErrors( cudaMalloc( ( void ** ) &synapseIndexMap.outgoingSynapseBegin, count * sizeof( BGSIZE ) ) );
  checkCudaErrors( cudaMalloc( ( void ** ) &synapseIndexMap.outgoingSynapseCount, count * sizeof( BGSIZE ) ) );
  checkCudaErrors( cudaMemset(synapseIndexMap.outgoingSynapseBegin, 0, count * sizeof( BGSIZE ) ) );
  checkCudaErrors( cudaMemset(synapseIndexMap.outgoingSynapseCount, 0, count * sizeof( BGSIZE ) ) );

  checkCudaErrors( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseBegin, count * sizeof( BGSIZE ) ) );
  checkCudaErrors( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseCount, count * sizeof( BGSIZE ) ) );
  checkCudaErrors( cudaMemset(synapseIndexMap.incomingSynapseBegin, 0, count * sizeof( BGSIZE ) ) );
  checkCudaErrors( cudaMemset(synapseIndexMap.incomingSynapseCount, 0, count * sizeof( BGSIZE ) ) );

  checkCudaErrors( cudaMalloc( ( void ** ) &m_synapseIndexMapDevice, sizeof( SynapseIndexMap ) ) );
  checkCudaErrors( cudaMemcpy( m_synapseIndexMapDevice, &synapseIndexMap, sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );
}

/*
 *  Deallocate device memory for synapse inverse map.
 */
void GPUSpikingCluster::deleteSynapseImap(  )
{
  SynapseIndexMap synapseIndexMap;

  checkCudaErrors( cudaMemcpy ( &synapseIndexMap, m_synapseIndexMapDevice, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );

  checkCudaErrors( cudaFree( synapseIndexMap.outgoingSynapseBegin ) );
  checkCudaErrors( cudaFree( synapseIndexMap.outgoingSynapseCount ) );
  checkCudaErrors( cudaFree( synapseIndexMap.outgoingSynapseIndexMap ) );

  checkCudaErrors( cudaFree( synapseIndexMap.incomingSynapseBegin ) );
  checkCudaErrors( cudaFree( synapseIndexMap.incomingSynapseCount ) );
  checkCudaErrors( cudaFree( synapseIndexMap.incomingSynapseIndexMap ) );

  checkCudaErrors( cudaFree( m_synapseIndexMapDevice ) );
}

/* 
 *  Copy SynapseIndexMap in host memory to SynapseIndexMap in device memory.
 *
 *  @param  clr_info    ClusterInfo to refer from.
 */
void GPUSpikingCluster::copySynapseIndexMapHostToDevice(const ClusterInfo *clr_info)
{
  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

  SynapseIndexMap *synapseIndexMapHost = m_synapseIndexMap;
  SynapseIndexMap *synapseIndexMapDevice = m_synapseIndexMapDevice;
  int total_synapse_counts = dynamic_cast<AllSynapses*>(m_synapses)->total_synapse_counts;
  int neuron_count = clr_info->totalClusterNeurons;

  if (synapseIndexMapHost == NULL || total_synapse_counts == 0)
    return;

  SynapseIndexMap synapseIndexMap;

  checkCudaErrors( cudaMemcpy ( &synapseIndexMap, synapseIndexMapDevice, sizeof( SynapseIndexMap ), cudaMemcpyDeviceToHost ) );

  // outgoing synaps index map
  checkCudaErrors( cudaMemcpy ( synapseIndexMap.outgoingSynapseBegin, synapseIndexMapHost->outgoingSynapseBegin, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  checkCudaErrors( cudaMemcpy ( synapseIndexMap.outgoingSynapseCount, synapseIndexMapHost->outgoingSynapseCount, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  // the number of synapses may change, so we reallocate the memory
  if (synapseIndexMap.outgoingSynapseIndexMap != NULL) {
    checkCudaErrors( cudaFree( synapseIndexMap.outgoingSynapseIndexMap ) );
  }
  checkCudaErrors( cudaMalloc( ( void ** ) &synapseIndexMap.outgoingSynapseIndexMap, total_synapse_counts * sizeof( OUTGOING_SYNAPSE_INDEX_TYPE ) ) );
  checkCudaErrors( cudaMemcpy ( synapseIndexMap.outgoingSynapseIndexMap, synapseIndexMapHost->outgoingSynapseIndexMap, total_synapse_counts * sizeof( OUTGOING_SYNAPSE_INDEX_TYPE ), cudaMemcpyHostToDevice ) );

  // incomming synapse index map
  checkCudaErrors( cudaMemcpy ( synapseIndexMap.incomingSynapseBegin, synapseIndexMapHost->incomingSynapseBegin, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  checkCudaErrors( cudaMemcpy ( synapseIndexMap.incomingSynapseCount, synapseIndexMapHost->incomingSynapseCount, neuron_count * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );
  // the number of synapses may change, so we reallocate the memory
  if (synapseIndexMap.incomingSynapseIndexMap != NULL) {
    checkCudaErrors( cudaFree( synapseIndexMap.incomingSynapseIndexMap ) );
  }
  checkCudaErrors( cudaMalloc( ( void ** ) &synapseIndexMap.incomingSynapseIndexMap, total_synapse_counts * sizeof( BGSIZE ) ) );
  checkCudaErrors( cudaMemcpy ( synapseIndexMap.incomingSynapseIndexMap, synapseIndexMapHost->incomingSynapseIndexMap, total_synapse_counts * sizeof( BGSIZE ), cudaMemcpyHostToDevice ) );

  checkCudaErrors( cudaMemcpy ( synapseIndexMapDevice, &synapseIndexMap, sizeof( SynapseIndexMap ), cudaMemcpyHostToDevice ) );
}

/* ------------------*\
   |* # Global Functions
   \* ------------------*/

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
__global__ void calcSummationMapDevice(int totalNeurons, 
				       AllSpikingNeuronsDeviceProperties* __restrict__ allNeuronsDevice, 
				       const SynapseIndexMap* __restrict__ synapseIndexMapDevice, 
				       const AllSpikingSynapsesDeviceProperties* __restrict__ allSynapsesDevice)
{
  // The usual thread ID calculation and guard against excess threads
  // (beyond the number of neurons, in this case).
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= totalNeurons )
    return;

  // Number of incoming synapses
  const BGSIZE synCount = synapseIndexMapDevice->incomingSynapseCount[idx];
  // Optimization: terminate thread if no incoming synapses
  if (synCount != 0) {
    // Index of start of this neuron's block of forward map entries
    const int beginIndex = synapseIndexMapDevice->incomingSynapseBegin[idx];
    // Address of the start of this neuron's block of forward map entries
    const BGSIZE* activeMap_begin = 
      &(synapseIndexMapDevice->incomingSynapseIndexMap[beginIndex]);
    // Summed postsynaptic response (PSR)
    BGFLOAT sum = 0.0;
    // Index of the current incoming synapse
    BGSIZE synIndex;
    // Repeat for each incoming synapse
    for (BGSIZE i = 0; i < synCount; i++) {
      // Get index of current incoming synapse
      synIndex = activeMap_begin[i];
      // Fetch its PSR and add into sum
      sum += allSynapsesDevice->psr[synIndex];
    }
    // Store summed PSR into this neuron's summation point
    allNeuronsDevice->summation_map[idx] = sum;
  }
}


