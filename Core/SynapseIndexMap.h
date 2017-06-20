/**
 *      @file SynapseIndexMap.h
 *
 *      @brief A structure maintains outgoing and active  synapses list (forward map).
 */

/**
 ** @struct SynapseIndexMap SynapseIndexMap.h "SynapseIndexMap.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** The strcuture maintains a list of outgoing synapses (forward map) and active synapses list.
 **
 ** The outgoing synapses list stores all outgoing synapse indexes relevant to a neuron.
 ** Synapse indexes are stored in the synapse forward map (forwardIndex), and
 ** the pointer and length of the neuron i's outgoing synapse indexes are specified 
 ** by outgoingSynapse_begin[i] and synapseCount[i] respectively. 
 ** The incoming synapses list is used in calcSummationMapDevice() device function to
 ** calculate sum of PSRs for each neuron simultaneously. 
 ** The list also used in AllSpikingNeurons::advanceNeurons() function to allow back propagation. 
 **
 ** The active synapses list stores all active synapse indexes. 
 ** The list is refered in advanceSynapsesDevice() device function.
 ** The list contribute to reduce the number of the device function thread to skip the inactive
 ** synapses.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other
 ** work (Stiber and Kawasaki (2007?))
 **
 **/

#pragma once
       
#include <stdint.h>
#include <vector>
#include "BGTypes.h"
#include "ClusterInfo.h"
#include "SimulationInfo.h"

class Cluster;

using namespace std;

class SynapseIndexMap
{
    public:
        SynapseIndexMap() : num_neurons(0), num_incoming_synapses(0), num_outgoing_synapses(0)
        {
            outgoingSynapseIndexMap = NULL;
            outgoingSynapseBegin = NULL;
            outgoingSynapseCount = NULL;

            incomingSynapseIndexMap = NULL;
            incomingSynapseBegin = NULL;
            incomingSynapseCount = NULL;
        };

        SynapseIndexMap(int neuron_count, BGSIZE synapse_count) : num_neurons(neuron_count), num_incoming_synapses(synapse_count)
        {
            outgoingSynapseBegin = new BGSIZE[neuron_count];
            outgoingSynapseCount = new BGSIZE[neuron_count];

            incomingSynapseIndexMap = new BGSIZE[synapse_count];
            incomingSynapseBegin = new BGSIZE[neuron_count];
            incomingSynapseCount = new BGSIZE[neuron_count];
        };

        ~SynapseIndexMap()
        {
            if (num_neurons != 0) {
                    delete[] outgoingSynapseBegin;
                    delete[] outgoingSynapseCount;
                    delete[] incomingSynapseBegin;
                    delete[] incomingSynapseCount;
            }
            if (num_incoming_synapses != 0) {
                    delete[] incomingSynapseIndexMap;
            }
            if (num_outgoing_synapses != 0) {
                    delete[] outgoingSynapseIndexMap;
            }
        }

        /**
         *  Get cluster index from outgoing synapse index.
         *
         *  @param  outSynIndex   Outgoing synapse index.
         */
        CUDA_CALLABLE static inline CLUSTER_INDEX_TYPE getClusterIndex(OUTGOING_SYNAPSE_INDEX_TYPE outSynIndex) {
            return outSynIndex >> CSC_SHIFT_COUNT; 
        };

        /**
         *  Get synapse index from outgoing synapse index.
         *
         *  @param  outSynIndex   Outgoing synapse index.
         */
        CUDA_CALLABLE static inline BGSIZE getSynapseIndex(OUTGOING_SYNAPSE_INDEX_TYPE outSynIndex) {
            return outSynIndex & SYNAPSE_INDEX_MASK;
        };

        /**
         *  Get outgoing synapse index from cluster and synapse index.
         *
         *  @param  idxCluster   Cluster index.
         *  @param  syn_i        Synapse index.
         */
        static inline OUTGOING_SYNAPSE_INDEX_TYPE getOutgoingSynapseIndex(CLUSTER_INDEX_TYPE idxCluster, BGSIZE syn_i) {
            return ((OUTGOING_SYNAPSE_INDEX_TYPE)idxCluster << CSC_SHIFT_COUNT) | syn_i;
        };

        /**
         *  Get cluster index from neuron layout index.
         *
         *  @param  iNeuron      Neuron layout index.
         *  @param  vtClrInfo    Vecttor of pointer to the ClusterInfo object.
         */
        static CLUSTER_INDEX_TYPE getClusterIdxFromNeuronLayoutIdx(int iNeuron, vector<ClusterInfo *> &vtClrInfo);

        /**
         *  Get internal neuron index in cluster from neuron layout index.
         *
         *  @param  iNeuron      Neuron layout index.
         *  @param  vtClrInfo    Vecttor of pointer to the ClusterInfo object.
         */
        static int getNeuronIdxFromNeuronLayoutIdx(int iNeuron, vector<ClusterInfo *> &vtClrInfo);

        /**
         *  Create a synapse index map.
         *
         *  @param  sim_info          Pointer to the simulation information.
         *  @param  vtClr             Vector of pointer to the Cluster object.
         *  @param  vtClrInfo         Vecttor of pointer to the ClusterInfo object.
         */
        static void createSynapseImap(const SimulationInfo* sim_info, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);

        /**
         *  Allocate memory for outgoing synapses index.
         *
         *  @param  synapse_count     Size for allocating memory.
         */
        void allocOutgoingSynapseIndexMap(BGSIZE synapse_count)
        {
            num_outgoing_synapses = synapse_count;
            outgoingSynapseIndexMap = new OUTGOING_SYNAPSE_INDEX_TYPE[synapse_count];
        };

    public:
        //! Pointer to the outgoing synapse index map.
        OUTGOING_SYNAPSE_INDEX_TYPE* outgoingSynapseIndexMap;

        //! The beginning index of the outgoing spiking synapse array of each neuron.
        //! Indexed by a source neuron index.
        BGSIZE* outgoingSynapseBegin;

        //! The array of number of outgoing synapses of each neuron.
        //! Indexed by a source neuron index.
        BGSIZE* outgoingSynapseCount;

        //! Pointer to the incoming synapse index map.
        BGSIZE* incomingSynapseIndexMap;

        //! The beginning index of the incoming spiking synapse array of each neuron.
        //! Indexed by a destination neuron index.
        BGSIZE* incomingSynapseBegin;

        //! The array of number of incoming synapses of each neuron.
        //! Indexed by a destination neuron index.
        BGSIZE* incomingSynapseCount;

    private:
        // Number of total neurons.
        BGSIZE num_neurons;

        // Number of total incoming synapses.
        BGSIZE num_incoming_synapses;

        // Number of total outging synapses.
        BGSIZE num_outgoing_synapses;
};

