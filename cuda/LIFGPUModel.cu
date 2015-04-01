#include "LIFGPUModel.h"

__global__ void setSynapseSummationPointDevice(int num_neurons, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice, int max_synapses, int width);

//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice );

//! Perform updating synapses for one time step.
__global__ void advanceSynapsesDevice ( int total_synapse_counts, GPUSpikingModel::SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllDSSynapses* allSynapsesDevice );

//! Calculate summation point.
__global__ void calcSummationMapDevice( int totalNeurons, GPUSpikingModel::SynapseIndexMap* synapseIndexMapDevice, AllDSSynapses* allSynapsesDevice );

//! Update the network.
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice );

//! Add a synapse to the network.
__device__ void addSynapse( AllDSSynapses* allSynapsesDevice, synapseType type, const int src_neuron, const int dest_neuron, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons );

//! Create a synapse.
__device__ void createSynapse( AllDSSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type );

//! Remove a synapse from the network.
__device__ void eraseSynapse( AllDSSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int maxSynapses );

//! Get the type of synapse.
__device__ synapseType synType( AllIFNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron );

//! Get the type of synapse (excitatory or inhibitory)
__device__ int synSign( synapseType t );

// ----------------------------------------------------------------------------

LIFGPUModel::LIFGPUModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) :
        GPUSpikingModel::GPUSpikingModel(conns, neurons, synapses, layout),
        m_allNeuronsDevice(NULL),
        m_allSynapsesDevice(NULL)
{
}

LIFGPUModel::~LIFGPUModel()
{
        //Let Model base class handle de-allocation
}

/**
 *  Sets up the Simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void LIFGPUModel::setupSim(SimulationInfo *sim_info, IRecorder* simRecorder)
{
    GPUSpikingModel::setupSim(sim_info, simRecorder);

    // allocates memories on CUDA device
    allocDeviceStruct((void **)&m_allNeuronsDevice, (void **)&m_allSynapsesDevice, sim_info);

    // set device summation points
    int neuron_count = sim_info->totalNeurons;
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;
    setSynapseSummationPointDevice <<< blocksPerGrid, threadsPerBlock >>> (neuron_count, m_allNeuronsDevice, m_allSynapsesDevice, sim_info->maxSynapsesPerNeuron, sim_info->width);
}

/**
 *  Loads the simulation based on istream input.
 *  @param  input   istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void LIFGPUModel::loadMemory(istream& input, const SimulationInfo *sim_info)
{
    GPUSpikingModel::loadMemory(input, sim_info);

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
 *  Notify outgoing synapses if neuron has fired.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFGPUModel::advanceNeurons(const SimulationInfo *sim_info)
{
    int neuron_count = sim_info->totalNeurons;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, g_simulationStep, sim_info->maxSynapsesPerNeuron, sim_info->deltaT, randNoise_d, m_allNeuronsDevice, m_allSynapsesDevice );
}

/**
 *  Advance all the Synapses in the simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 */
void LIFGPUModel::advanceSynapses(const SimulationInfo *sim_info)
{
    size_t total_synapse_counts = m_synapses->total_synapse_counts;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( total_synapse_counts + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance synapses ------------->
    advanceSynapsesDevice <<< blocksPerGrid, threadsPerBlock >>> ( total_synapse_counts, synapseIndexMapDevice, g_simulationStep, sim_info->deltaT, m_allSynapsesDevice );
}

void LIFGPUModel::calcSummationMap(const SimulationInfo *sim_info)
{
    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
 
    calcSummationMapDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, synapseIndexMapDevice, m_allSynapsesDevice );
}

/** 
*  Begin terminating the simulator.
*  @param  sim_info    SimulationInfo to refer.
*/
void LIFGPUModel::cleanupSim(SimulationInfo *sim_info)
{
    // copy device synapse and neuron structs to host memory
    m_neurons->copyNeuronDeviceToHost( m_allNeuronsDevice, sim_info );
    m_synapses->copySynapseDeviceToHost( m_allSynapsesDevice, sim_info );

    // Deallocate device memory
    m_neurons->deleteNeuronDeviceStruct(m_allNeuronsDevice, sim_info);
    m_synapses->deleteSynapseDeviceStruct(m_allSynapsesDevice, sim_info);

    GPUSpikingModel::cleanupSim(sim_info);
}

/**
 *  Get synapse_counts in AllSynapses struct on device memory.
 *  @param  allSynapsesHost     Reference to the AllSynapses struct on host memory.
 *  @param  neuron_coun         The number of neurons.
 */
void LIFGPUModel::copyDeviceSynapseCountsToHost(AllSynapses &allSynapsesHost, int neuron_count)
{
        AllDSSynapses allSynapses;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses, m_allSynapsesDevice, sizeof( AllDSSynapses ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.synapse_counts, allSynapses.synapse_counts, neuron_count * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
}

/** 
 *  Get summationCoord and in_use in AllSynapses struct on device memory.
 *  @param  allSynapsesHost     Reference to the AllSynapses struct on host memory.
 *  @param  neuron_coun         The number of neurons.
 *  @param  max_synapses        Maximum number of synapses per neuron.
 */
void LIFGPUModel::copyDeviceSynapseSumCoordToHost(AllSynapses &allSynapsesHost, int neuron_count, int max_synapses)
{
        AllDSSynapses allSynapses_0;

        HANDLE_ERROR( cudaMemcpy ( &allSynapses_0, m_allSynapsesDevice, sizeof( AllDSSynapses ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.summationCoord, allSynapses_0.summationCoord,
                max_synapses * neuron_count * sizeof( Coordinate ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allSynapsesHost.in_use, allSynapses_0.in_use,
                max_synapses * neuron_count * sizeof( bool ), cudaMemcpyDeviceToHost ) );
}

/**
 *  Get spike history in AllIFNeurons struct on device memory.
 *  @param  allNeuronsHost      Reference to the allNeurons struct on host memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void LIFGPUModel::copyDeviceSpikeHistoryToHost(AllSpikingNeurons &allNeuronsHost, const SimulationInfo *sim_info)
{
        AllIFNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, m_allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );

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
 *  Get spikeCount in AllIFNeurons struct on device memory.
 *  @param  allNeuronsHost      Reference to the allNeurons struct on host memory.
 *  @param  numNeurons          The number of neurons.
 */
void LIFGPUModel::copyDeviceSpikeCountsToHost(AllSpikingNeurons &allNeuronsHost, int numNeurons)
{
        AllIFNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, m_allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.spikeCount, allNeurons.spikeCount, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/** 
*  Clear the spike counts out of all Neurons.
*  @param  numNeurons The number of neurons.
*/
void LIFGPUModel::clearSpikeCounts(int numNeurons)
{
        AllIFNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, m_allNeuronsDevice, sizeof( AllIFNeurons ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemset( allNeurons.spikeCount, 0, numNeurons * sizeof( int ) ) );
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

        blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
        updateNetworkDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, width, deltaT, W_d, sim_info->maxSynapsesPerNeuron, m_allNeuronsDevice, m_allSynapsesDevice );

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

/**
 * Set the summation points in device memory
 * @param[in] num_neurons        Number of neurons.
 * @param[in] allNeuronsDevice   Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesDevice  Pointer to the Synapse structures in device memory.
 * @param[in] max_synapses       Maximum number of synapses per neuron.
 * @param[in] width              Width of neuron map (assumes square).
 */
__global__ void setSynapseSummationPointDevice(int num_neurons, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice, int max_synapses, int width)
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

// CUDA code for advancing neurons
/**
* @param[in] totalNeurons       Number of neurons.
* @param[in] simulationStep     The current simulation step.
* @param[in] maxSynapses        Maximum number of synapses per neuron.
* @param[in] deltaT             Inner simulation step duration.
* @param[in] randNoise          Pointer to device random noise array.
* @param[in] allNeuronsDevice   Pointer to Neuron structures in device memory.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice ) {
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
* @param[in] total_synapse_counts       Total number of synapses.
* @param[in] synapseIndexMap            Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
* @param[in] simulationStep             The current simulation step.
* @param[in] deltaT                     Inner simulation step duration.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void advanceSynapsesDevice ( int total_synapse_counts, GPUSpikingModel::SynapseIndexMap* synapseIndexMapDevice, uint64_t simulationStep, const BGFLOAT deltaT, AllDSSynapses* allSynapsesDevice ) {
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
* @param[in] totalNeurons       Number of neurons.
* @param[in] synapseIndexMap    Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
* @param[in] allSynapsesDevice  Pointer to Synapse structures in device memory.
*/
__global__ void calcSummationMapDevice( int totalNeurons, GPUSpikingModel::SynapseIndexMap* synapseIndexMapDevice, AllDSSynapses* allSynapsesDevice ) {
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
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllIFNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice )
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
* @param neuron_index   Index of a neuron.
* @param synapse_index  Index of a synapse.
* @param[in] maxSynapses        Maximum number of synapses per neuron.
*/
__device__ void eraseSynapse( AllDSSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int maxSynapses )
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
__device__ void addSynapse(AllDSSynapses* allSynapsesDevice, synapseType type, const int src_neuron, const int dest_neuron, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons)
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
    createSynapse(allSynapsesDevice, src_neuron, synapse_index, source_x, source_y, dest_x, dest_y, sum_point, deltaT, type );
    allSynapsesDevice->W[iSync + synapse_index] = W_d[src_neuron * num_neurons + dest_neuron] * synSign(type) * SYNAPSE_STRENGTH_ADJUSTMENT;
}

/**
 *  Create a Synapse and connect it to the model.
 *  @param allSynapsesDevice    Pointer to the Synapse structures in device memory.
 *  @param neuron_index         Index of the source neuron.
 *  @param synapse_index        Index of the Synapse to create.
 *  @param source_x             X location of source.
 *  @param source_y             Y location of source.
 *  @param dest_x               X location of destination.
 *  @param dest_y               Y location of destination.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createSynapse(AllDSSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    size_t max_synapses = allSynapsesDevice->maxSynapsesPerNeuron;
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
* @param src_neuron             Index of the source neuron.
* @param dest_neuron            Index of the destination neuron.
*/
__device__ synapseType synType( AllIFNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron )
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
 
