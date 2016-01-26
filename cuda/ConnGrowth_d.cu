#include "ConnGrowth.h"
#include "AllSpikingSynapses.h"
#include "Book.h"

/*
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *  @param  num_neurons number of neurons to update.
 *  @param  neurons the Neuron list to search from.
 *  @param  synapses    the Synapse list to search from.
 *  @param  sim_info    SimulationInfo to refer from.
 */
void ConnGrowth::updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeurons* m_allNeuronsDevice, AllSpikingSynapses* m_allSynapsesDevice, Layout *layout)
{
        // For now, we just set the weights to equal the areas. We will later
        // scale it and set its sign (when we index and get its sign).
        (*W) = (*area);

        BGFLOAT deltaT = sim_info->deltaT;

        // CUDA parameters
        const int threadsPerBlock = 256;
        int blocksPerGrid;

        // allocate device memories
        size_t W_d_size = sim_info->totalNeurons * sim_info->totalNeurons * sizeof (BGFLOAT);
        BGFLOAT* W_h = new BGFLOAT[W_d_size];
        BGFLOAT* W_d;
        HANDLE_ERROR( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

        neuronType* neuron_type_map_d;
        HANDLE_ERROR( cudaMalloc( ( void ** ) &neuron_type_map_d, sim_info->totalNeurons * sizeof( neuronType ) ) );

        // copy weight data to the device memory
        for ( int i = 0 ; i < sim_info->totalNeurons; i++ )
                for ( int j = 0; j < sim_info->totalNeurons; j++ )
                        W_h[i * sim_info->totalNeurons + j] = (*W)(i, j);

        HANDLE_ERROR( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

        HANDLE_ERROR( cudaMemcpy ( neuron_type_map_d, layout->neuron_type_map, sim_info->totalNeurons * sizeof( neuronType ), cudaMemcpyHostToDevice ) );

        unsigned long long fpCreateSynapse_h;
        synapses.getFpCreateSynapse(fpCreateSynapse_h);

        blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
        updateSynapsesWeightsDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, deltaT, W_d, sim_info->maxSynapsesPerNeuron, m_allNeuronsDevice, m_allSynapsesDevice, (void (*)(AllSpikingSynapses*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType))fpCreateSynapse_h, neuron_type_map_d );

        // free memories
        HANDLE_ERROR( cudaFree( W_d ) );
        delete[] W_h;

        HANDLE_ERROR( cudaFree( neuron_type_map_d ) );

        // copy device synapse count to host memory
        synapses.copyDeviceSynapseCountsToHost(m_allSynapsesDevice, sim_info);
        // copy device synapse summation coordinate to host memory
        synapses.copyDeviceSynapseSumIdxToHost(m_allSynapsesDevice, sim_info);
}

/*
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below
 * zero.
 * @param[in] num_neurons        Number of neurons.
 * @param[in] deltaT             The time step size.
 * @param[in] W_d                Array of synapse weight.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsDevice          Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesDevice         Pointer to the Synapse structures in device memory.
 */
__global__ void updateSynapsesWeightsDevice( int num_neurons, BGFLOAT deltaT, BGFLOAT* W_d, int maxSynapses, AllSpikingNeurons* allNeuronsDevice, AllSpikingSynapses* allSynapsesDevice, void (*fpCreateSynapse)(AllSpikingSynapses*, const int, const int, int, int, BGFLOAT*, const BGFLOAT, synapseType), neuronType* neuron_type_map_d )
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

    // and each destination neuron 'b'
    for (int dest_neuron = 0; dest_neuron < num_neurons; dest_neuron++) {
        // visit each synapse at (xa,ya)
        bool connected = false;
        synapseType type = synType(neuron_type_map_d, src_neuron, dest_neuron);

        // for each existing synapse
        size_t synapse_counts = allSynapsesDevice->synapse_counts[src_neuron];
        int synapse_adjusted = 0;
        for (size_t synapse_index = 0; synapse_adjusted < synapse_counts; synapse_index++) {
            uint32_t iSyn = maxSynapses * src_neuron + synapse_index;
            if (allSynapsesDevice->in_use[iSyn] == true) {
                // if there is a synapse between a and b
                if (allSynapsesDevice->destNeuronIndex[iSyn] == dest_neuron) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.
                    if (W_d[src_neuron * num_neurons + dest_neuron] < 0) {
                        removed++;
                        eraseSpikingSynapse(allSynapsesDevice, src_neuron, synapse_index, maxSynapses);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesDevice->W[iSyn] = W_d[src_neuron * num_neurons
                            + dest_neuron] * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
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

            addSpikingSynapse(allSynapsesDevice, type, src_neuron, dest_neuron, src_neuron, dest_neuron, sum_point, deltaT, W_d, num_neurons, fpCreateSynapse);
        }
    }
}
