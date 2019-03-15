#include "AllSynapsesDeviceFuncs.h"
#include "AllSynapses.h"
#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapses.h"
#include "math_constants.h"
#include "HeapSort.hpp"


// a device variable to store synapse class ID.
__device__ enumClassSynapses classSynapses_d = classAllDSSynapses; 
//__device__ enumClassSynapses classSynapses_d = undefClassSynapses; 

/* ------------------------------------*\
|* # Device Functions for createSynapse
\* ------------------------------------*/

/*
 * Return 1 if originating neuron is excitatory, -1 otherwise.
 *
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

/*
 *  Create a Spiking Synapse and connect it to the model.
 *
 *  @param allSynapsesProps    Pointer to the AllSpikingSynapsesProps structures 
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createSpikingSynapse(AllSpikingSynapsesProps* allSynapsesProps, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesProps->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesProps->in_use[iSyn] = true;
    allSynapsesProps->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesProps->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesProps->W[iSyn] = synSign(type) * 10.0e-9;
    
    allSynapsesProps->psr[iSyn] = 0.0;
    allSynapsesProps->type[iSyn] = type;

    allSynapsesProps->tau[iSyn] = DEFAULT_tau;

    BGFLOAT tau;
    switch (type) {
        case II:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesProps->tau[iSyn] = tau;
    allSynapsesProps->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesProps->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    assert( allSynapsesProps->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY );

    // initializes the queues for the Synapses
    allSynapsesProps->preSpikeQueue->clearAnEvent(iSyn);
}

/*
 *  Create a DS Synapse and connect it to the model.
 *
 *  @param allSynapsesProps    Pointer to the AllSpikingSynapsesProps structures 
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createDSSynapse(AllDSSynapsesProps* allSynapsesProps, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesProps->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesProps->in_use[iSyn] = true;
    allSynapsesProps->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesProps->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesProps->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesProps->psr[iSyn] = 0.0;
    allSynapsesProps->r[iSyn] = 1.0;
    allSynapsesProps->u[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesProps->lastSpike[iSyn] = ULONG_MAX;
    allSynapsesProps->type[iSyn] = type;

    allSynapsesProps->U[iSyn] = DEFAULT_U;
    allSynapsesProps->tau[iSyn] = DEFAULT_tau;

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

    allSynapsesProps->U[iSyn] = U;
    allSynapsesProps->D[iSyn] = D;
    allSynapsesProps->F[iSyn] = F;

    allSynapsesProps->tau[iSyn] = tau;
    allSynapsesProps->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesProps->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    assert( allSynapsesProps->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY );

    // initializes the queues for the Synapses
    allSynapsesProps->preSpikeQueue->clearAnEvent(iSyn);
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param allSynapsesProps    Pointer to the AllSpikingSynapsesProps structures 
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createSTDPSynapse(AllSTDPSynapsesProps* allSynapsesProps, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesProps->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesProps->in_use[iSyn] = true;
    allSynapsesProps->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesProps->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesProps->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesProps->psr[iSyn] = 0.0;
    allSynapsesProps->type[iSyn] = type;

    allSynapsesProps->tau[iSyn] = DEFAULT_tau;

    BGFLOAT tau;
    switch (type) {
        case II:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case IE:
            tau = 6e-3;
            delay = 0.8e-3;
            break;
        case EI:
            tau = 3e-3;
            delay = 0.8e-3;
            break;
        case EE:
            tau = 3e-3;
            delay = 1.5e-3;
            break;
        default:
            break;
    }

    allSynapsesProps->tau[iSyn] = tau;
    allSynapsesProps->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesProps->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    assert( allSynapsesProps->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY );

    allSynapsesProps->Apos[iSyn] = 0.5;
    allSynapsesProps->Aneg[iSyn] = -0.5;
    allSynapsesProps->STDPgap[iSyn] = 2e-3;

    allSynapsesProps->total_delayPost[iSyn] = 0;

    allSynapsesProps->tauspost[iSyn] = 0;
    allSynapsesProps->tauspre[iSyn] = 0;

    allSynapsesProps->taupos[iSyn] = 15e-3;
    allSynapsesProps->tauneg[iSyn] = 35e-3;
    allSynapsesProps->Wex[iSyn] = 1.0;

    allSynapsesProps->mupos[iSyn] = 0;
    allSynapsesProps->muneg[iSyn] = 0;

    allSynapsesProps->useFroemkeDanSTDP[iSyn] = false;

    // initializes the queues for the Synapses
    allSynapsesProps->postSpikeQueue->clearAnEvent(iSyn);
}

/*
 *  Create a Synapse and connect it to the model.
 *
 *  @param allSynapsesProps    Pointer to the AllSpikingSynapsesProps structures 
 *                              on device memory.
 *  @param neuron_index         Index of the destination neuron in the cluster.
 *  @param synapse_offset       Offset (into neuron_index's) of the Synapse to create.
 *  @param source_index         Layout index of the source neuron.
 *  @param dest_index           Layout index of the destination neuron.
 *  @param sum_point            Pointer to the summation point.
 *  @param deltaT               The time step size.
 *  @param type                 Type of the Synapse to create.
 */
__device__ void createDynamicSTDPSynapse(AllDynamicSTDPSynapsesProps* allSynapsesProps, const int neuron_index, const int synapse_offset, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, synapseType type)
{
    BGFLOAT delay;
    BGSIZE max_synapses = allSynapsesProps->maxSynapsesPerNeuron;
    BGSIZE iSyn = max_synapses * neuron_index + synapse_offset;

    allSynapsesProps->in_use[iSyn] = true;
    allSynapsesProps->destNeuronLayoutIndex[iSyn] = dest_index;
    allSynapsesProps->sourceNeuronLayoutIndex[iSyn] = source_index;
    allSynapsesProps->W[iSyn] = synSign(type) * 10.0e-9;

    allSynapsesProps->psr[iSyn] = 0.0;
    allSynapsesProps->r[iSyn] = 1.0;
    allSynapsesProps->u[iSyn] = 0.4;     // DEFAULT_U
    allSynapsesProps->lastSpike[iSyn] = ULONG_MAX;
    allSynapsesProps->type[iSyn] = type;

    allSynapsesProps->U[iSyn] = DEFAULT_U;
    allSynapsesProps->tau[iSyn] = DEFAULT_tau;

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

    allSynapsesProps->U[iSyn] = U;
    allSynapsesProps->D[iSyn] = D;
    allSynapsesProps->F[iSyn] = F;

    allSynapsesProps->tau[iSyn] = tau;
    allSynapsesProps->decay[iSyn] = exp( -deltaT / tau );
    allSynapsesProps->total_delay[iSyn] = static_cast<int>( delay / deltaT ) + 1;

    assert( allSynapsesProps->total_delay[iSyn] >= MIN_SYNAPTIC_TRANS_DELAY );

    allSynapsesProps->Apos[iSyn] = 0.5;
    allSynapsesProps->Aneg[iSyn] = -0.5;
    allSynapsesProps->STDPgap[iSyn] = 2e-3;

    allSynapsesProps->total_delayPost[iSyn] = 0;

    allSynapsesProps->tauspost[iSyn] = 0;
    allSynapsesProps->tauspre[iSyn] = 0;

    allSynapsesProps->taupos[iSyn] = 15e-3;
    allSynapsesProps->tauneg[iSyn] = 35e-3;
    allSynapsesProps->Wex[iSyn] = 1.0;

    allSynapsesProps->mupos[iSyn] = 0;
    allSynapsesProps->muneg[iSyn] = 0;

    allSynapsesProps->useFroemkeDanSTDP[iSyn] = false;

    // initializes the queues for the Synapses
    allSynapsesProps->postSpikeQueue->clearAnEvent(iSyn);
}

/*
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 *
 * @param allSynapsesProps      Pointer to the AllSpikingSynapsesProps structures 
 *                               on device memory.
 * @param type                   Type of the Synapse to create.
 * @param neuron_index           Index of the destination neuron in the cluster.
 * @param source_index           Layout index of the source neuron.
 * @param dest_index             Layout index of the destination neuron.
 * @param sum_point              Pointer to the summation point.
 * @param deltaT                 The time step size.
 * @param weight                 Synapse weight.
 */
__device__ void addSpikingSynapse(AllSpikingSynapsesProps* allSynapsesProps, synapseType type, const int neuron_index, int source_index, int dest_index, BGFLOAT *sum_point, const BGFLOAT deltaT, BGFLOAT weight)
{
    if (allSynapsesProps->synapse_counts[neuron_index] >= allSynapsesProps->maxSynapsesPerNeuron) {
        assert(false);
        return; // TODO: ERROR!
    }

    // add it to the list
    BGSIZE synapse_index;
    BGSIZE max_synapses = allSynapsesProps->maxSynapsesPerNeuron;
    BGSIZE synapseBegin = max_synapses * neuron_index;
    for (synapse_index = 0; synapse_index < max_synapses; synapse_index++) {
        if (!allSynapsesProps->in_use[synapseBegin + synapse_index]) {
            break;
        }
    }

    allSynapsesProps->synapse_counts[neuron_index]++;

    // create a synapse
    switch (classSynapses_d) {
    case classAllSpikingSynapses:
        createSpikingSynapse(allSynapsesProps, neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    case classAllDSSynapses:
        createDSSynapse(static_cast<AllDSSynapsesProps *>(allSynapsesProps), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    case classAllSTDPSynapses:
        createSTDPSynapse(static_cast<AllSTDPSynapsesProps *>(allSynapsesProps), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    case classAllDynamicSTDPSynapses:
        createDynamicSTDPSynapse(static_cast<AllDynamicSTDPSynapsesProps *>(allSynapsesProps), neuron_index, synapse_index, source_index, dest_index, sum_point, deltaT, type );
        break;
    default:
        assert(false);
    }
    allSynapsesProps->W[synapseBegin + synapse_index] = weight;
}

/*
 * Remove a synapse from the network.
 *
 * @param[in] allSynapsesProps      Pointer to the AllSpikingSynapsesProps structures 
 *                                   on device memory.
 * @param neuron_index               Index of the destination neuron in the cluster.
 * @param synapse_offset             Offset into neuron_index's synapses.
 * @param[in] maxSynapses            Maximum number of synapses per neuron.
 */
__device__ void eraseSpikingSynapse( AllSpikingSynapsesProps* allSynapsesProps, const int neuron_index, const int synapse_offset, int maxSynapses )
{
    BGSIZE iSync = maxSynapses * neuron_index + synapse_offset;
    allSynapsesProps->synapse_counts[neuron_index]--;
    allSynapsesProps->in_use[iSync] = false;
}

/*
 * Returns the type of synapse at the given coordinates
 *
 * @param[in] allNeuronsProps          Pointer to the Neuron structures in device memory.
 * @param src_neuron             Index of the source neuron.
 * @param dest_neuron            Index of the destination neuron.
 */
__device__ synapseType synType( neuronType* neuron_type_map_d, const int src_neuron, const int dest_neuron )
{
    if ( neuron_type_map_d[src_neuron] == INH && neuron_type_map_d[dest_neuron] == INH )
        return II;
    else if ( neuron_type_map_d[src_neuron] == INH && neuron_type_map_d[dest_neuron] == EXC )
        return IE;
    else if ( neuron_type_map_d[src_neuron] == EXC && neuron_type_map_d[dest_neuron] == INH )
        return EI;
    else if ( neuron_type_map_d[src_neuron] == EXC && neuron_type_map_d[dest_neuron] == EXC )
        return EE;

    return STYPE_UNDEF;

}

/* -------------------------------------*\
|* # Global Functions for updateSynapses
\* -------------------------------------*/

/*
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below
 * zero.
 *
 * @param[in] num_neurons        Total number of neurons.
 * @param[in] deltaT             The time step size.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsProps   Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesProps  Pointer to the Synapse structures in device memory.
 * @param[in] neuron_type_map_d   Pointer to the neurons type map in device memory.
 * @param[in] totalClusterNeurons  Total number of neurons in the cluster.
 * @param[in] clusterNeuronsBegin  Begin neuron index of the cluster.
 * @param[in] radii_d            Pointer to the rates data array.
 * @param[in] xloc_d             Pointer to the neuron's x location array.
 * @param[in] yloc_d             Pointer to the neuron's y location array.
 */
__global__ void updateSynapsesWeightsDevice( int num_neurons, BGFLOAT deltaT, int maxSynapses, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, neuronType* neuron_type_map_d, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* radii_d, BGFLOAT* xloc_d,  BGFLOAT* yloc_d )
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
        synapseType type = synType(neuron_type_map_d, src_neuron, dest_neuron);

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
                        eraseSpikingSynapse(allSynapsesProps, iNeuron, synapse_index, maxSynapses);
                    } else {
                        // adjust
                        // g_synapseStrengthAdjustmentConstant is 1.0e-8;
                        allSynapsesProps->W[iSyn] = area * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
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

            BGFLOAT weight = area * synSign(type) * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
            addSpikingSynapse(allSynapsesProps, type, iNeuron, src_neuron, dest_neuron, sum_point, deltaT, weight);
        }
    }
}

/*
 *  CUDA kernel function for setting up connections.
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters:
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
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
 *  @param  allNeuronsProps    Pointer to the Neuron structures in device memory.
 *  @param  allSynapsesProps   Pointer to the Synapse structures in device memory.
 *  @param  minExcWeight        Min values of excitatory neuron's synapse weight.
 *  @param  maxExcWeight        Max values of excitatory neuron's synapse weight.
 *  @param  minInhWeight        Min values of inhibitory neuron's synapse weight.
 *  @param  maxInhWeight        Max values of inhibitory neuron's synapse weight.
 *  @param  devStates_d         Curand global state.
 *  @param  seed                Seed for curand.
 */
__global__ void setupConnectionsDevice( int num_neurons, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* xloc_d, BGFLOAT* yloc_d, int nConnsPerNeuron, int threshConnsRadius, neuronType* neuron_type_map_d, ConnStatic::DistDestNeuron *rDistDestNeuron_d, BGFLOAT deltaT, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, BGFLOAT minExcWeight, BGFLOAT maxExcWeight, BGFLOAT minInhWeight, BGFLOAT maxInhWeight, curandState* devStates_d, unsigned long seed )
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
        synapseType type = synType(neuron_type_map_d, src_neuron, dest_neuron);

        // create a synapse at the cluster of the destination neuron

        DEBUG_MID ( printf("source: %d dest: %d dist: %d\n", src_neuron, dest_neuron, distDestNeuron.dist); )

        // set synapse weight
        // TODO: we need another synaptic weight distibution mode (normal distribution)
        BGFLOAT weight;
        curandState localState = devStates_d[iNeuron];
        if (synSign(type) > 0) {
            weight = minExcWeight + curand_uniform( &localState ) * (maxExcWeight - minExcWeight);
        }
        else {
            weight = minInhWeight + curand_uniform( &localState ) * (maxInhWeight - minInhWeight);
        }
        devStates_d[iNeuron] = localState;

        BGFLOAT* sum_point = &( allNeuronsProps->summation_map[iNeuron] );
        addSpikingSynapse(allSynapsesProps, type, iNeuron, src_neuron, dest_neuron, sum_point, deltaT, weight);

    }
}

/* 
 * Adds a synapse to the network.  Requires the locations of the source and
 * destination neurons.
 *
 * @param allSynapsesProps      Pointer to the Synapse structures in device memory.
 * @param pSummationMap          Pointer to the summation point.
 * @param width                  Width of neuron map (assumes square).
 * @param deltaT                 The simulation time step size.
 * @param weight                 Synapse weight.
 */
__global__ void initSynapsesDevice( int n, AllDSSynapsesProps* allSynapsesProps, BGFLOAT *pSummationMap, int width, const BGFLOAT deltaT, BGFLOAT weight )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n )
        return;

    // create a synapse
    int neuron_index = idx;
    BGFLOAT* sum_point = &( pSummationMap[neuron_index] );
    synapseType type = allSynapsesProps->type[neuron_index];
    createDSSynapse(allSynapsesProps, neuron_index, 0, 0, neuron_index, sum_point, deltaT, type );
    allSynapsesProps->W[neuron_index] = weight * AllSynapses::SYNAPSE_STRENGTH_ADJUSTMENT;
}

