#include "IZHGPUModel.h"

__global__ void setSynapseSummationPointDevice(int num_neurons, AllIZHNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice, int max_synapses, int width);

//! Perform updating neurons for one time step.
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllIZHNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice );

//! Update the network.
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllIZHNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice );

//! Add a synapse to the network.
extern __device__ void addSynapse( AllDSSynapses* allSynapsesDevice, synapseType type, const int src_neuron, const int dest_neuron, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT* W_d, int num_neurons );

//! Create a synapse.
extern __device__ void createSynapse( AllDSSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int source_x, int source_y, int dest_x, int dest_y, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type );

//! Remove a synapse from the network.
extern __device__ void eraseSynapse( AllDSSynapses* allSynapsesDevice, const int neuron_index, const int synapse_index, int maxSynapses );

//! Get the type of synapse.
__device__ synapseType synType( AllIZHNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron );

//! Get the type of synapse (excitatory or inhibitory)
extern __device__ int synSign( synapseType t );

// ----------------------------------------------------------------------------

IZHGPUModel::IZHGPUModel(Connections *conns, AllNeurons *neurons, AllSynapses *synapses, Layout *layout) :
        GPUSpikingModel::GPUSpikingModel(conns, neurons, synapses, layout),
        m_allNeuronsDevice(NULL)
{
}

IZHGPUModel::~IZHGPUModel()
{
        //Let Model base class handle de-allocation
}

/**
 *  Sets up the Simulation.
 *  @param  sim_info    SimulationInfo class to read information from.
 *  @param  simRecorder Pointer to the simulation recordig object.
 */
void IZHGPUModel::setupSim(SimulationInfo *sim_info, IRecorder* simRecorder)
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
*  Begin terminating the simulator.
*  @param  sim_info    SimulationInfo to refer.
*/
void IZHGPUModel::cleanupSim(SimulationInfo *sim_info)
{
    // deallocates memories on CUDA device
    deleteDeviceStruct((void**)&m_allNeuronsDevice, (void**)&m_allSynapsesDevice, sim_info);

    GPUSpikingModel::cleanupSim(sim_info);
}

/**
 *  Loads the simulation based on istream input.
 *  @param  input   istream to read from.
 *  @param  sim_info    used as a reference to set info for neurons and synapses.
 */
void IZHGPUModel::loadMemory(istream& input, const SimulationInfo *sim_info)
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
void IZHGPUModel::advanceNeurons(const SimulationInfo *sim_info)
{
    int neuron_count = sim_info->totalNeurons;

    // CUDA parameters
    const int threadsPerBlock = 256;
    int blocksPerGrid = ( neuron_count + threadsPerBlock - 1 ) / threadsPerBlock;

    // Advance neurons ------------->
    advanceNeuronsDevice <<< blocksPerGrid, threadsPerBlock >>> ( neuron_count, g_simulationStep, sim_info->maxSynapsesPerNeuron, sim_info->deltaT, randNoise_d, m_allNeuronsDevice, m_allSynapsesDevice );
}

/**
 *  Get spike history in AllIZHNeurons struct on device memory.
 *  @param  allNeuronsHost      Reference to the allNeurons struct on host memory.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void IZHGPUModel::copyDeviceSpikeHistoryToHost(AllSpikingNeurons &allNeuronsHost, const SimulationInfo *sim_info)
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, m_allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );

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
 *  Get spikeCount in AllIZHNeurons struct on device memory.
 *  @param  allNeuronsHost      Reference to the allNeurons struct on host memory.
 *  @param  numNeurons          The number of neurons.
 */
void IZHGPUModel::copyDeviceSpikeCountsToHost(AllSpikingNeurons &allNeuronsHost, int numNeurons)
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, m_allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy ( allNeuronsHost.spikeCount, allNeurons.spikeCount, numNeurons * sizeof( int ), cudaMemcpyDeviceToHost ) );
}

/** 
*  Clear the spike counts out of all Neurons.
*  @param  numNeurons The number of neurons.
*/
void IZHGPUModel::clearSpikeCounts(int numNeurons)
{
        AllIZHNeurons allNeurons;
        HANDLE_ERROR( cudaMemcpy ( &allNeurons, m_allNeuronsDevice, sizeof( AllIZHNeurons ), cudaMemcpyDeviceToHost ) );
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
void IZHGPUModel::updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo *sim_info)
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
__global__ void setSynapseSummationPointDevice(int num_neurons, AllIZHNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice, int max_synapses, int width)
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
__global__ void advanceNeuronsDevice( int totalNeurons, uint64_t simulationStep, int maxSynapses, const BGFLOAT deltaT, float* randNoise, AllIZHNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice ) {
        // determine which neuron this thread is processing
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ( idx >= totalNeurons )
                return;

        allNeuronsDevice->hasFired[idx] = false;
        BGFLOAT& sp = allNeuronsDevice->summation_map[idx];
        BGFLOAT& vm = allNeuronsDevice->Vm[idx];
        BGFLOAT& a = allNeuronsDevice->Aconst[idx];
        BGFLOAT& b = allNeuronsDevice->Bconst[idx];
        BGFLOAT& u = allNeuronsDevice->u[idx];
        BGFLOAT r_sp = sp;
        BGFLOAT r_vm = vm;
        BGFLOAT r_a = a;
        BGFLOAT r_b = b;
        BGFLOAT r_u = u;

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
                vm = allNeuronsDevice->Cconst[idx] * 0.001;
                u = r_u + allNeuronsDevice->Dconst[idx];

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

                BGFLOAT Vint = r_vm * 1000;

                // Izhikevich model integration step
                BGFLOAT Vb = Vint + allNeuronsDevice->C3[idx] * (0.04 * Vint * Vint + 5 * Vint + 140 - u);
                u = r_u + allNeuronsDevice->C3[idx] * r_a * (r_b * Vint - r_u);

                vm = Vb * 0.001 + allNeuronsDevice->C2[idx] * r_sp;  // add inputs
        }

        // clear synaptic input for next time step
        sp = 0;
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
__global__ void updateNetworkDevice( int num_neurons, int width, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllIZHNeurons* allNeuronsDevice, AllDSSynapses* allSynapsesDevice )
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
* Returns the type of synapse at the given coordinates
* @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
* @param src_neuron             Index of the source neuron.
* @param dest_neuron            Index of the destination neuron.
*/
__device__ synapseType synType( AllIZHNeurons* allNeuronsDevice, const int src_neuron, const int dest_neuron )
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

