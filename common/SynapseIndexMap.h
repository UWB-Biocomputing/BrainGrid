/**
 *      @file SynapseIndexMap.h
 *
 *      @brief A structure maintains incoming synapses list (inverse map).
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
        
        struct SynapseIndexMap
        {
                //! The beginning index of the incoming dynamic spiking synapse array.
                int* outgoingSynapse_begin;

                //! The array of number of active synapses of each neuron.
                int* synapseCount;

                //! Pointer to the synapse inverse map.
                uint32_t* forwardIndex;

                //! Pointer to the active synapse map.
                uint32_t* activeSynapseIndex;

                SynapseIndexMap() : num_neurons(0), num_synapses(0)
                {
                        outgoingSynapse_begin = NULL;
                        synapseCount = NULL;
                        forwardIndex = NULL;
                        activeSynapseIndex = NULL;
                };

                SynapseIndexMap(int neuron_count, int synapse_count) : num_neurons(neuron_count), num_synapses(synapse_count)
                {
                        outgoingSynapse_begin = new int[neuron_count];
                        synapseCount = new int[neuron_count];
                        forwardIndex = new uint32_t[synapse_count];
                        activeSynapseIndex = new uint32_t[synapse_count];
                };

                ~SynapseIndexMap()
                {
                        if (num_neurons != 0) {
                                delete[] outgoingSynapse_begin;
                                delete[] synapseCount;
                        }
                        if (num_synapses != 0) {
                                delete[] forwardIndex;
                                delete[] activeSynapseIndex;
                        }
                }

        private:
                // Number of total neurons.
                int num_neurons;

                // Number of total active synapses.
                int num_synapses;
        };

