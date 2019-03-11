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
#include "MersenneTwister_d.h"

// ----------------------------------------------------------------------------

#if defined(VALIDATION)
// Buffer to save random numbers in host memory
float* GPUSpikingCluster::m_randNoiseHost;
#endif // VALIDATION

GPUSpikingCluster::GPUSpikingCluster(IAllNeurons *neurons, IAllSynapses *synapses) : 	
  Cluster::Cluster(neurons, synapses),
  m_synapseIndexMapDevice(NULL),
  randNoise_d(NULL),
  m_allNeuronsDeviceProps(NULL),
  m_allSynapsesDeviceProps(NULL)
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
  DEBUG(
    {
        // Report GPU memory usage
        printf("\n");

        size_t free_byte;
        size_t total_byte;

        checkCudaErrors( cudaMemGetInfo( &free_byte, &total_byte ) );

        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db = total_db - free_db;

        printf("Before allocating GPU memories\n");
        printf("GPU memory usage: device ID = %d, used = %5.3f MB, free = %5.3f MB, total = %5.3f MB\n", clr_info->deviceId, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

        printf("\n");
    }
  ) // end  DEBUG

  // Allocate Neurons and Synapses strucs on GPU device memory
  AllNeuronsProps *pNeuronsProps = dynamic_cast<AllNeurons*>(m_neurons)->m_pNeuronsProps;
  pNeuronsProps->setupNeuronsDeviceProps( allNeuronsDevice, sim_info, clr_info );

  AllSynapsesProps *pSynapsesProps = dynamic_cast<AllSynapses*>(m_synapses)->m_pSynapsesProps;
  pSynapsesProps->setupSynapsesDeviceProps( allSynapsesDevice, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );

  // Copy host neuron and synapse arrays into GPU device
  pNeuronsProps->copyNeuronHostToDeviceProps( *allNeuronsDevice, sim_info, clr_info );
  pSynapsesProps->copySynapseHostToDeviceProps( *allSynapsesDevice, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );

  // allocate synapse index map in device memory
  int neuron_count = clr_info->totalClusterNeurons;
  allocSynapseImap( neuron_count );

#if defined(VALIDATION)
  // allocate buffer to save random numbers in host memory
  if (clr_info->clusterID == 0) {
    BGSIZE randNoiseBufferSize = sizeof(float) * sim_info->totalNeurons * sim_info->minSynapticTransDelay;
    m_randNoiseHost = new float[randNoiseBufferSize];
  }
#endif // VALIDATION

  DEBUG(
    {
        // Report GPU memory usage
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
  ) // end  DEBUG
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
  AllNeuronsProps *pNeuronsProps = dynamic_cast<AllNeurons*>(m_neurons)->m_pNeuronsProps;
  pNeuronsProps->copyNeuronDeviceToHostProps( *allNeuronsDevice, sim_info, clr_info );

  // Deallocate device memory
  pNeuronsProps->cleanupNeuronsDeviceProps( *allNeuronsDevice, clr_info );

  // copy device synapse and neuron structs to host memory
  AllSynapsesProps *pSynapsesProps = dynamic_cast<AllSynapses*>(m_synapses)->m_pSynapsesProps;
  pSynapsesProps->copySynapseDeviceToHostProps( *allSynapsesDevice, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );

  // Deallocate device memory
  pSynapsesProps->cleanupSynapsesDeviceProps( *allSynapsesDevice );

  deleteSynapseImap();

#if defined(VALIDATION)
  // Deallocate buffer to save random numbers in host memory
  if (clr_info->clusterID == 0) {
    delete[] m_randNoiseHost;
  }
#endif // VALIDATION

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

  // initialize Mersenne Twister
 
#if defined(VALIDATION)
  BGSIZE randNoise_d_size;
  if (clr_info->clusterID == 0) {
    //assuming neuron_count >= 100 and is a multiple of 100. Note rng_mt_rng_count must be <= MT_RNG_COUNT
    int rng_blocks = 25; //# of blocks the kernel will use
    int rng_nPerRng = 4; //# of iterations per thread (thread granularity, # of rands generated per thread)
    int rng_mt_rng_count = sim_info->totalNeurons / rng_nPerRng; //# of threads to generate for neuron_count rand #s
    int rng_threads = rng_mt_rng_count/rng_blocks; //# threads per block needed

    initMTGPU(clr_info->seed, rng_blocks, rng_threads, rng_nPerRng, rng_mt_rng_count);

    // Allocate memory for random noise array
    randNoise_d_size = rng_mt_rng_count * rng_nPerRng * sizeof (float) * sim_info->minSynapticTransDelay;	// size of random noise array
    checkCudaErrors( cudaMalloc ( ( void ** ) &randNoise_d, randNoise_d_size ) );
  } else {
    randNoise_d_size = clr_info->totalClusterNeurons * sizeof (float);	// size of random noise array
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

  // allocates memories on CUDA device
  allocDeviceStruct((void **)&m_allNeuronsDeviceProps, (void **)&m_allSynapsesDeviceProps, sim_info, clr_info);

  // create an AllNeurons class object in device
  m_neurons->createAllNeuronsInDevice(&m_neuronsDevice, m_allNeuronsDeviceProps);

  // set some parameters used for advanceNeuronsDevice
  m_neurons->setAdvanceNeuronsDeviceParams(*m_synapses);

  // create an AllSynapses class object in device
  m_synapses->createAllSynapsesInDevice(&m_synapsesDevice, m_allSynapsesDeviceProps);

  // set some parameters used for advanceSynapsesDevice
  m_synapses->setAdvanceSynapsesDeviceParams();

  // assign an address of summation function

  // NOTE: When the number of synapses per neuron is smaller, parallel reduction
  // method exhibits poor performance. The coefficient K is a tentative value,
  // and needs to be adjusted.
  // We may use another way to choose better kernel based on the measurement of
  // real execution time of each kernel. 

  // sequential addtion base summation
  clr_info->fpCalcSummationMap = &GPUSpikingCluster::calcSummationMap_2;

#if 0
  // Disabled parallel reduction base summatio kernel. 
  // In most case, sequential addtion base summation kernel exhibits better 
  // performance.
  BGFLOAT K = 1.0;
  if (sim_info->maxSynapsesPerNeuron * K > clr_info->totalClusterNeurons) {
    // parallel reduction base summation
    clr_info->fpCalcSummationMap = &GPUSpikingCluster::calcSummationMap_1;
  } else {
    // sequential addtion base summation
    clr_info->fpCalcSummationMap = &GPUSpikingCluster::calcSummationMap_2;
  }
#endif
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
  deleteDeviceStruct((void**)&m_allNeuronsDeviceProps, (void**)&m_allSynapsesDeviceProps, sim_info, clr_info);

  // delete an AllNeurons class object in device
  m_neurons->deleteAllNeuronsInDevice(m_neuronsDevice);

  // delete an AllSynapses class object in device
  m_synapses->deleteAllSynapsesInDevice(m_synapsesDevice);

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
  AllNeuronsProps *pNeuronsProps = dynamic_cast<AllNeurons*>(m_neurons)->m_pNeuronsProps;
  pNeuronsProps->copyNeuronHostToDeviceProps( m_allNeuronsDeviceProps, sim_info, clr_info );

  AllSynapsesProps *pSynapsesProps = dynamic_cast<AllSynapses*>(m_synapses)->m_pSynapsesProps;
  pSynapsesProps->copySynapseHostToDeviceProps( m_allSynapsesDeviceProps, clr_info->totalClusterNeurons, sim_info->maxSynapsesPerNeuron );
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
  // generates random numbers in cluster 0
  if (clr_info->clusterID != 0) {
    return;
  }

  // Set device ID
  checkCudaErrors( cudaSetDevice( clr_info->deviceId ) );

#ifdef PERFORMANCE_METRICS
  // Reset CUDA timer to start measurement of GPU operation
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // generates random numbers for all clusters within transmission delay
  for (int i = 0; i < m_nSynapticTransDelay; i++) {
    normalMTGPU(randNoise_d + sim_info->totalNeurons * i);
  }

  // and copy them to host memory to share among clusters
  BGSIZE randNoiseBufferSize = sizeof (float) * sim_info->totalNeurons * m_nSynapticTransDelay;
  checkCudaErrors( cudaMemcpy ( m_randNoiseHost, randNoise_d, randNoiseBufferSize,  cudaMemcpyDeviceToHost ) );

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_rndGeneration);
#endif // PERFORMANCE_METRICS
}
#endif // VALIDATION

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

  // Advance neurons ------------->
  m_neurons->advanceNeurons(*m_synapses, m_allNeuronsDeviceProps, m_allSynapsesDeviceProps, sim_info, randNoiseDevice, m_synapseIndexMapDevice, clr_info, iStepOffset, m_neuronsDevice);

#else // !VALIDATION

  normalMTGPU(randNoise_d);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_rndGeneration);
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // Advance neurons ------------->
  m_neurons->advanceNeurons(*m_synapses, m_allNeuronsDeviceProps, m_allSynapsesDeviceProps, sim_info, randNoise_d, m_synapseIndexMapDevice, clr_info, iStepOffset, m_neuronsDevice);

#endif // !VALIDATION

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

#ifdef PERFORMANCE_METRICS
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // process inter clusters outgoing spikes
  AllSynapses* pSynapses = dynamic_cast<AllSynapses*>(m_synapses);
  dynamic_cast<AllSpikingSynapsesProps*>(pSynapses->m_pSynapsesProps)->processInterClustesOutgoingSpikes(m_allSynapsesDeviceProps);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_processInterClustesOutgoingSpikes);
#endif // PERFORMANCE_METRICS
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

#ifdef PERFORMANCE_METRICS
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // process inter clusters incoming spikes
  AllSynapses* pSynapses = dynamic_cast<AllSynapses*>(m_synapses);
  dynamic_cast<AllSpikingSynapsesProps*>(pSynapses->m_pSynapsesProps)->processInterClustesIncomingSpikes(m_allSynapsesDeviceProps);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_processInterClustesIncomingSpikes);
#endif // PERFORMANCE_METRICS
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
  m_synapses->advanceSynapses(m_allSynapsesDeviceProps, m_allNeuronsDeviceProps, m_synapseIndexMapDevice, sim_info, clr_info, iStepOffset, m_synapsesDevice, m_neuronsDevice);

#ifdef PERFORMANCE_METRICS
  cudaLapTime(clr_info, clr_info->t_gpu_advanceSynapses);
  cudaStartTimer(clr_info);
#endif // PERFORMANCE_METRICS

  // wait until all CUDA related tasks complete
  checkCudaErrors( cudaDeviceSynchronize() );

  // calculate summation point
  (this->*(clr_info->fpCalcSummationMap))(sim_info, clr_info);

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

  (dynamic_cast<AllSpikingSynapses*>(m_synapses))->advanceSpikeQueue(m_allSynapsesDeviceProps, iStep);

  if (sim_info->pInput != NULL) {
      // advance input stimulus state
      sim_info->pInput->advanceSInputState(clr_info, iStep);
  }

  // wait until all CUDA related tasks complete
  checkCudaErrors( cudaDeviceSynchronize() );
}

/*
 * Add psr of all incoming synapses to summation points.
 * (parallel reduction base summation)
 *
 * @param[in] sim_info                   Pointer to the simulation information.
 * @param[in] clr_info                   Pointer to the cluster information.
 */
void GPUSpikingCluster::calcSummationMap_1(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
  if (m_synapseIndexMap == NULL) {
    return;
  }

  BGSIZE numTotalSynapses = m_synapseIndexMap->num_incoming_synapses;

  if (numTotalSynapses == 0) {
    return;
  }

  // call parallel reduction base summation kernel
  calcSummationMapDevice_1 <<< 1, 1 >>> (numTotalSynapses, m_allNeuronsDeviceProps, m_synapseIndexMapDevice, m_allSynapsesDeviceProps, sim_info->maxSynapsesPerNeuron, clr_info->clusterNeuronsBegin);
}

/*
 * Add psr of all incoming synapses to summation points.
 * (sequential addtion base summation)
 *
 * @param[in] sim_info                   Pointer to the simulation information.
 * @param[in] clr_info                   Pointer to the cluster information.
 */
void GPUSpikingCluster::calcSummationMap_2(const SimulationInfo *sim_info, const ClusterInfo *clr_info)
{
  if (m_synapseIndexMap == NULL) {
    return;
  }

  BGSIZE numTotalSynapses = m_synapseIndexMap->num_incoming_synapses;

  if (numTotalSynapses == 0) {
    return;
  }

  // CUDA parameters
  const int threadsPerBlock = 256;
  int blocksPerGrid = ( clr_info->totalClusterNeurons + threadsPerBlock - 1 ) / threadsPerBlock;

  // call sequential addtion base summation kernel
  calcSummationMapDevice_2 <<< blocksPerGrid, threadsPerBlock >>> ( clr_info->totalClusterNeurons, m_allNeuronsDeviceProps, m_synapseIndexMapDevice, m_allSynapsesDeviceProps );
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
  int total_synapse_counts = dynamic_cast<AllSynapses*>(m_synapses)->m_pSynapsesProps->total_synapse_counts;
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

  DEBUG(
        {
            // Report GPU memory usage
            printf("\n");

            size_t free_byte;
            size_t total_byte;

            checkCudaErrors( cudaMemGetInfo( &free_byte, &total_byte ) );

            double free_db = (double)free_byte;
            double total_db = (double)total_byte;
            double used_db = total_db - free_db;

            printf("After creating SynapseIndexMap\n");
            printf("GPU memory usage: device ID = %d, used = %5.3f MB, free = %5.3f MB, total = %5.3f MB\n", clr_info->deviceId, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

            printf("\n");
        }
  ) // end  DEBUG
}

   /* ------------------*\
   |* # Global Functions
   \* ------------------*/

/**
 * Calculate the sum of synaptic input to each neuron.
 * (use parallel reduction method)
 *
 * Calculate the sum of synaptic input to each neuron's block. This kernel
 * spawns another kernel function (reduceSummationMapKernel), which 
 * corresponfs each incoming synapse, to perform  parallel reduction 
 * method to calculate the summation of synaptic inputs (dynamic 
 * parallelism in CUDA).
 *
 * @param[in] numTotalSynapses       Number of total incoming synapses.
 * @param[in,out] allNeuronsProps   Pointer to Neuron structures in device memory.
 * @param[in] synapseIndexMapDevice  Pointer to synapse index  map structures in device memory.
 * @param[in] allSynapsesProps  Pointer to Synapse structures in device memory.
 * @param[in] maxSynapsesPerNeuron   Maximum number of synapses per neuron. 
 * @param[in] clusterNeuronsBegin    Start neuron index of the cluster.
 */
__global__ void calcSummationMapDevice_1(BGSIZE numTotalSynapses, AllSpikingNeuronsProps* allNeuronsProps, SynapseIndexMap* synapseIndexMapDevice, AllSpikingSynapsesProps* allSynapsesProps, int maxSynapsesPerNeuron, int clusterNeuronsBegin)
{
  // CUDA  parameters
  const int threadsPerBlock = 256;
  int blocksPerGrid = ( numTotalSynapses + threadsPerBlock - 1 ) / threadsPerBlock;

  // pointer to the incoming synapses index map
  BGSIZE* indexMap = synapseIndexMapDevice->incomingSynapseIndexMap;

  // pointer to the count of each neuron's block of incoming map entries
  BGSIZE* synapseCount = synapseIndexMapDevice->incomingSynapseCount;

  // pointer to the start of each neuron's block of incoming map entries
  BGSIZE* synapseBegin = synapseIndexMapDevice->incomingSynapseBegin;

  // do reduction
  for (unsigned int s = 1; s < maxSynapsesPerNeuron; s *= 2) {
    // Call another CUDA kernel (dynamic parallelism in CUDA) here.
    // Because CUDA does not have global thread synchronization, we need to invoke
    // another CUDA kernel function to synchronize reduction steps.

    reduceSummationMapKernel <<< blocksPerGrid, threadsPerBlock >>> (numTotalSynapses, s, allSynapsesProps, allNeuronsProps, indexMap, synapseCount, synapseBegin, clusterNeuronsBegin);
  }
}

/**
 * Helper kernel function for calcSummationMapDevice.
 *
 * Calculate the sum of synaptic input to each neuron's block. 
 * One thread corresponfs each incoming synapse and performs one step of 
 * parallel reduction method to calculate the summation of synaptic inputs.
 *
 * @param[in] numTotalSynapses      Number of total incoming synapses.
 * @param[in] s                     Size of stride of the reduction.
 * @param[in] allSynapsesProps     Pointer to Synapse structures in device memory.
 * @param[in,out] allNeuronsProps  Pointer to Neuron structures in device memory.
 * @param[in] indexMap              Pointer to incoming synapses index map.
 * @param[in] synapseCount          Pointer to count of each neuron's block of incoming map entries.
 * @param[in] synapseBegin          Pointer to start of each neuron's block of incoming map entries.
 * @param[in] clusterNeuronsBegin   Start neuron index of the cluster.
 */
__global__ void reduceSummationMapKernel(BGSIZE numTotalSynapses, unsigned int s, AllSpikingSynapsesProps* allSynapsesProps, AllSpikingNeuronsProps* allNeuronsProps, BGSIZE* indexMap, BGSIZE* synapseCount, BGSIZE* synapseBegin, int clusterNeuronsBegin)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ( idx >= numTotalSynapses )
    return;

  // Add two 'psr' (post synapti response) values of synapses, indexed by 'synIndex_1' and
  // 'synIndex_2' (synIndex_1+s), and stores the result to 'summation' of the first synapse.
  // At end of reduction, the 'summation' has the total 'psr' summation of each neuron's block.
  // 'indexMap' stores all incoming synapse indexes. Synapses that have the same destination
  // neuron (same 'destNeuronLayoutIndex') belong to the same neuron's bloack, and 
  // are stored sequencially in the 'indexMap'. Start index and number of synapses of each 
  // neuron's block (specified by 'neuIndex_1') are identfied by 'synapseBegin[neuIndex_1]' 
  // and 'synapseCount[neuIndex_1]' respectively.
  // 'destNeuronLayoutIndex' is global neuron index, and 'neuIndex_1' is local (cluster)
  // neuron index, so we need conversion (subtracting clusterNeuronsBegin).

  BGSIZE synIndex_1 = indexMap[idx]; 
  int neuIndex_1 = allSynapsesProps->destNeuronLayoutIndex[synIndex_1] - clusterNeuronsBegin;
  BGSIZE synCount = synapseCount[neuIndex_1];
  if (s > synCount)
    return;

  BGSIZE beginIndex = synapseBegin[neuIndex_1];
  int offsetIndex = idx - beginIndex;
  DEBUG_MID( assert( offsetIndex >= 0 ); )

  if (offsetIndex % (2*s) == 0) {
    if (s == 1) {
      allSynapsesProps->summation[synIndex_1] = allSynapsesProps->psr[synIndex_1];
    }

    if (offsetIndex + s < synCount ) {
      BGSIZE synIndex_2 = indexMap[idx + s];

      DEBUG_MID(
      int neuIndex_2 = allSynapsesProps->destNeuronLayoutIndex[synIndex_2] - clusterNeuronsBegin;
      assert( neuIndex_1 == neuIndex_2 );
      ) // end DEBUG

      if (s == 1) {
        allSynapsesProps->summation[synIndex_2] = allSynapsesProps->psr[synIndex_2];
      }

      allSynapsesProps->summation[synIndex_1] = allSynapsesProps->summation[synIndex_1] + allSynapsesProps->summation[synIndex_2];
    }

    if ( (2*s) >= synCount) {
      allNeuronsProps->summation_map[neuIndex_1] = allSynapsesProps->summation[synIndex_1];
    }
  }
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
 * @param[in,out] allNeuronsProps   Pointer to Neuron structures in device memory.
 * @param[in] synapseIndexMapDevice  Pointer to forward map structures in device memory.
 * @param[in] allSynapsesDevice      Pointer to Synapse structures in device memory.
 */
__global__ void calcSummationMapDevice_2(int totalNeurons, 
				       AllSpikingNeuronsProps* __restrict__ allNeuronsProps, 
				       const SynapseIndexMap* __restrict__ synapseIndexMapDevice, 
				       const AllSpikingSynapsesProps* __restrict__ allSynapsesProps)
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
      sum += allSynapsesProps->psr[synIndex];
    }
    // Store summed PSR into this neuron's summation point
    allNeuronsProps->summation_map[idx] = sum;
  }
}
