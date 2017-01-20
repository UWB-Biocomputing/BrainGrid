#include "ConnGrowth.h"
#include "AllSpikingSynapses.h"
#include "AllSynapsesDeviceFuncs.h"
#include <helper_cuda.h>

/*
 *  Update the weight of the Synapses in the simulation.
 *  Note: Platform Dependent.
 *
 *  @param  num_neurons         number of neurons to update.
 *  @param  neurons             the Neuron list to search from.
 *  @param  synapses            the Synapse list to search from.
 *  @param  sim_info            SimulationInfo to refer from.
 *  @param  m_allNeuronsDevice  Reference to the allNeurons struct on device memory. 
 *  @param  m_allSynapsesDevice Reference to the allSynapses struct on device memory.
 *  @param  layout              Layout information of the neunal network.
 *  @paam   clr_info            ClusterInfo to refer from.
 */
void ConnGrowth::updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeuronsDeviceProperties* m_allNeuronsDevice, AllSpikingSynapsesDeviceProperties* m_allSynapsesDevice, Layout *layout, const ClusterInfo *clr_info)
{
        // For now, we just set the weights to equal the areas. We will later
        // scale it and set its sign (when we index and get its sign).
        (*W) = (*area);

        BGFLOAT deltaT = sim_info->deltaT;

        // CUDA parameters
        const int threadsPerBlock = 256;
        int blocksPerGrid;

        // allocate device memories
        BGSIZE W_d_size = sim_info->totalNeurons * sim_info->totalNeurons * sizeof (BGFLOAT);
        BGFLOAT* W_h = new BGFLOAT[W_d_size];
        BGFLOAT* W_d;
        checkCudaErrors( cudaMalloc ( ( void ** ) &W_d, W_d_size ) );

        neuronType* neuron_type_map_d;
        checkCudaErrors( cudaMalloc( ( void ** ) &neuron_type_map_d, sim_info->totalNeurons * sizeof( neuronType ) ) );

        // copy weight data to the device memory
        for ( int i = 0 ; i < sim_info->totalNeurons; i++ )
                for ( int j = 0; j < sim_info->totalNeurons; j++ )
                        W_h[i * sim_info->totalNeurons + j] = (*W)(i, j);

        checkCudaErrors( cudaMemcpy ( W_d, W_h, W_d_size, cudaMemcpyHostToDevice ) );

        checkCudaErrors( cudaMemcpy ( neuron_type_map_d, layout->neuron_type_map, sim_info->totalNeurons * sizeof( neuronType ), cudaMemcpyHostToDevice ) );

        blocksPerGrid = ( sim_info->totalNeurons + threadsPerBlock - 1 ) / threadsPerBlock;
        updateSynapsesWeightsDevice <<< blocksPerGrid, threadsPerBlock >>> ( sim_info->totalNeurons, deltaT, W_d, sim_info->maxSynapsesPerNeuron, m_allNeuronsDevice, m_allSynapsesDevice, neuron_type_map_d );

        // free memories
        checkCudaErrors( cudaFree( W_d ) );
        delete[] W_h;

        checkCudaErrors( cudaFree( neuron_type_map_d ) );

        // copy device synapse count to host memory
        synapses.copyDeviceSynapseCountsToHost(m_allSynapsesDevice, clr_info);
        // copy device synapse summation coordinate to host memory
        synapses.copyDeviceSynapseSumIdxToHost(m_allSynapsesDevice, sim_info, clr_info);
}
