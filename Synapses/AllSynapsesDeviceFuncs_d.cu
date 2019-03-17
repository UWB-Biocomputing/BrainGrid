#include "AllSynapsesDeviceFuncs.h"
#include "AllSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "math_constants.h"
#include "HeapSort.hpp"

/* -------------------------------------*\
|* # Global Functions for updateSynapses
\* -------------------------------------*/

/*
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below
 * zero.
 *
 * @param[in] synapsesDevice     Pointer to the Synapses object in device memory.
 * @param[in] num_neurons        Total number of neurons.
 * @param[in] deltaT             The time step size.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsProps    Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesProps   Pointer to the Synapse structures in device memory.
 * @param[in] neuron_type_map_d    Pointer to the neurons type map in device memory.
 * @param[in] totalClusterNeurons  Total number of neurons in the cluster.
 * @param[in] clusterNeuronsBegin  Begin neuron index of the cluster.
 * @param[in] radii_d            Pointer to the rates data array.
 * @param[in] xloc_d             Pointer to the neuron's x location array.
 * @param[in] yloc_d             Pointer to the neuron's y location array.
 */
__global__ void updateSynapsesWeightsDevice( IAllSynapses* synapsesDevice, int num_neurons, BGFLOAT deltaT, int maxSynapses, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, neuronType* neuron_type_map_d, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* radii_d, BGFLOAT* xloc_d,  BGFLOAT* yloc_d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= totalClusterNeurons )
        return;

    int adjusted = 0;
    int removed = 0;
    int added = 0;

    int iNeuron = idx;
    int dest_neuron = clusterNeuronsBegin + idx;

    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        if (dest_neuron == src_neuron) {
            // we don't create a synapse between the same neuron.
            continue;
        }

        // Update the areas of overlap in between Neurons
        BGFLOAT distX = xloc_d[dest_neuron] - xloc_d[src_neuron];
        BGFLOAT distY = yloc_d[dest_neuron] - yloc_d[src_neuron];
        BGFLOAT dist2 = distX * distX + distY * distY;
        BGFLOAT dist = sqrt(dist2);
        BGFLOAT delta = dist - (radii_d[dest_neuron] + radii_d[src_neuron]);
        BGFLOAT area = 0.0;

        if (delta < 0) {
            BGFLOAT lenAB = dist;
            BGFLOAT r1 = radii_d[dest_neuron];
            BGFLOAT r2 = radii_d[src_neuron];

            if (lenAB + min(r1, r2) <= max(r1, r2)) {
                area = CUDART_PI_F * min(r1, r2) * min(r1, r2); // Completely overlapping unit
            } else {
                // Partially overlapping unit
                BGFLOAT lenAB2 = dist2;
                BGFLOAT r12 = r1 * r1;
                BGFLOAT r22 = r2 * r2;

                BGFLOAT cosCBA = (r22 + lenAB2 - r12) / (2.0 * r2 * lenAB);
                BGFLOAT angCBA = acos(cosCBA);
                BGFLOAT angCBD = 2.0 * angCBA;

                BGFLOAT cosCAB = (r12 + lenAB2 - r22) / (2.0 * r1 * lenAB);
                BGFLOAT angCAB = acos(cosCAB);
                BGFLOAT angCAD = 2.0 * angCAB;

                area = 0.5 * (r22 * (angCBD - sin(angCBD)) + r12 * (angCAD - sin(angCAD)));
            }
        }

        // visit each synapse at (xa,ya)
        bool connected = false;
        synapseType type = synapsesDevice->synType(neuron_type_map_d, src_neuron, dest_neuron);

        // for each existing synapse
        BGSIZE existing_synapses = allSynapsesProps->synapse_counts[iNeuron];
        int existing_synapses_checked = 0;
        for (BGSIZE synapse_index = 0; (existing_synapses_checked < existing_synapses) && !connected; synapse_index++) {
            BGSIZE iSyn = maxSynapses * iNeuron + synapse_index;
            if (allSynapsesProps->in_use[iSyn] == true) {
                // if there is a synapse between a and b
                if (allSynapsesProps->sourceNeuronLayoutIndex[iSyn] == src_neuron) {
                    connected = true;
                    adjusted++;

                    // adjust the strength of the synapse or remove
                    // it from the synapse map if it has gone below
                    // zero.

                    // W_d[] is indexed by (dest_neuron (local index) * totalNeurons + src_neuron)
                    if (area < 0) {
                        removed++;
                        synapsesDevice->eraseSynapse(iNeuron, iSyn);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesProps->W[iSyn] = area * synapsesDevice->synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
                    }
                }
                existing_synapses_checked++;
            }
        }

        // if not connected and weight(a,b) > 0, add a new synapse from a to b
        if (!connected && (area > 0)) {
            // locate summation point
            BGFLOAT* sum_point = &( allNeuronsProps->summation_map[iNeuron] );
            added++;

            BGSIZE iSyn;
            synapsesDevice->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, deltaT, iNeuron);
            BGFLOAT weight = area * synapsesDevice->synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
            allSynapsesProps->W[iSyn] = weight;
        }
    }
}

/*
 *  CUDA kernel function for setting up connections.
 *iNeuron  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters:
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param  synapsesDevice      Pointer to the Synapses object in device memory.
 *  @param  num_neurons         Number of total neurons.
 *  @param  totalClusterNeurons	Total number of neurons in the cluster.
 *  @param  clusterNeuronsBegin Begin neuron index of the cluster.
 *  @param  xloc_d              Pointer to the neuron's x location array.
 *  @param  yloc_d              Pointer to the neuron's y location array.
 *  @param  nConnsPerNeuron     Number of maximum connections per neurons.
 *  @param  threshConnsRadius   Connection radius threshold.
 *  @param  neuron_type_map_d   Pointer to the neurons type map in device memory.
 *  @param  rDistDestNeuron_d   Pointer to the DistDestNeuron structure array.
 *  @param  deltaT              The time step size.
 *  @param  allNeuronsProps     Pointer to the Neuron structures in device memory.
 *  @param  allSynapsesProps    Pointer to the Synapse structures in device memory.
 *  @param  minExcWeight        Min values of excitatory neuron's synapse weight.
 *  @param  maxExcWeight        Max values of excitatory neuron's synapse weight.
 *  @param  minInhWeight        Min values of inhibitory neuron's synapse weight.
 *  @param  maxInhWeight        Max values of inhibitory neuron's synapse weight.
 *  @param  devStates_d         Curand global state.
 *  @param  seed                Seed for curand.
 */
__global__ void setupConnectionsDevice( IAllSynapses* synapsesDevice, int num_neurons, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* xloc_d, BGFLOAT* yloc_d, int nConnsPerNeuron, int threshConnsRadius, neuronType* neuron_type_map_d, ConnStatic::DistDestNeuron *rDistDestNeuron_d, BGFLOAT deltaT, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, BGFLOAT minExcWeight, BGFLOAT maxExcWeight, BGFLOAT minInhWeight, BGFLOAT maxInhWeight, curandState* devStates_d, unsigned long seed )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= totalClusterNeurons )
        return;

    int iNeuron = idx;
    int dest_neuron = iNeuron + clusterNeuronsBegin;

    // pick the connections shorter than threshConnsRadius
    BGSIZE iArrayBegin = num_neurons * iNeuron, iArrayEnd, iArray;
    iArray = iArrayBegin;
    for (int src_neuron = 0; src_neuron < num_neurons; src_neuron++) {
        if (src_neuron != dest_neuron) {
            BGFLOAT distX = xloc_d[dest_neuron] - xloc_d[src_neuron];
            BGFLOAT distY = yloc_d[dest_neuron] - yloc_d[src_neuron];
            BGFLOAT dist2 = distX * distX + distY * distY;
            BGFLOAT dist = sqrt(dist2);

            if (dist <= threshConnsRadius) {
                ConnStatic::DistDestNeuron distDestNeuron;
                distDestNeuron.dist = dist;
                distDestNeuron.src_neuron = src_neuron;
                rDistDestNeuron_d[iArray++] = distDestNeuron;
            }
        }
    }


    // sort ascendant
    iArrayEnd = iArray;
    int size = iArrayEnd - iArrayBegin;
    // CUDA thrust sort consumes heap memory, and when sorting large contents
    // it may cause an error "temporary_buffer::allocate: get_temporary_buffer failed".
    // Therefore we use local implementation of heap sort.
    // NOTE: Heap sort is an in-palce algoprithm (memory requirement is 1).
    // Its implementation is not stable. Time complexity is O(n*logn).
    heapSort(&rDistDestNeuron_d[iArrayBegin], size);

    // set up an initial state for curand
    curand_init( seed, iNeuron, 0, &devStates_d[iNeuron] );

    // pick the shortest nConnsPerNeuron connections
    iArray = iArrayBegin;
    for (BGSIZE i = 0; iArray < iArrayEnd && (int)i < nConnsPerNeuron; iArray++, i++) {
        ConnStatic::DistDestNeuron distDestNeuron = rDistDestNeuron_d[iArray];
        int src_neuron = distDestNeuron.src_neuron;
        synapseType type = synapsesDevice->synType(neuron_type_map_d, src_neuron, dest_neuron);

        // create a synapse at the cluster of the destination neuron

        DEBUG_MID ( printf("source: %d dest: %d dist: %d\n", src_neuron, dest_neuron, distDestNeuron.dist); )

        // set synapse weight
        // TODO: we need another synaptic weight distibution mode (normal distribution)
        BGFLOAT weight;
        curandState localState = devStates_d[iNeuron];
        if (synapsesDevice->synSign(type) > 0) {
            weight = minExcWeight + curand_uniform( &localState ) * (maxExcWeight - minExcWeight);
        }
        else {
            weight = minInhWeight + curand_uniform( &localState ) * (maxInhWeight - minInhWeight);
        }
        devStates_d[iNeuron] = localState;

        BGFLOAT* sum_point = &( allNeuronsProps->summation_map[iNeuron] );
        BGSIZE iSyn;
        synapsesDevice->addSynapse(iSyn, type, src_neuron, dest_neuron, sum_point, deltaT, iNeuron);
        allSynapsesProps->W[iSyn] = weight;

    }
}

/* 
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 *
 * @param synapsesDevice         Pointer to the Synapses object in device memory.
 * @param allSynapsesProps       Pointer to the Synapse structures in device memory.
 * @param pSummationMap          Pointer to the summation point.
 * @param width                  Width of neuron map (assumes square).
 * @param deltaT                 The simulation time step size.
 * @param weight                 Synapse weight.
 */
__global__ void initSynapsesDevice( IAllSynapses* synapsesDevice, int n, AllDSSynapsesProps* allSynapsesProps, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    // create a synapse
    int neuron_index = idx;
    BGFLOAT* sum_point = &( pSummationMap[neuron_index] );
    synapseType type = allSynapsesProps->type[neuron_index];

    BGSIZE iSyn = allSynapsesProps->maxSynapsesPerNeuron * neuron_index;
    synapsesDevice->createSynapse(iSyn, 0, neuron_index, sum_point, deltaT, type);
    allSynapsesProps->W[neuron_index] = weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
}

