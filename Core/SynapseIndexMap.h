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
        
        struct SynapseIndexMap
        {
                //! Pointer to the outgoing synapse index map.
                BGSIZE* outgoingSynapseIndexMap;

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

                SynapseIndexMap() : num_neurons(0), num_synapses(0)
                {
                        outgoingSynapseIndexMap = NULL;
                        outgoingSynapseBegin = NULL;
                        outgoingSynapseCount = NULL;

                        incomingSynapseIndexMap = NULL;
                        incomingSynapseBegin = NULL;
                        incomingSynapseCount = NULL;
                };

                SynapseIndexMap(int neuron_count, int synapse_count) : num_neurons(neuron_count), num_synapses(synapse_count)
                {
                        outgoingSynapseIndexMap = new BGSIZE[synapse_count];
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
                        if (num_synapses != 0) {
                                delete[] outgoingSynapseIndexMap;
                                delete[] incomingSynapseIndexMap;
                        }
                }

        private:
                // Number of total neurons.
                BGSIZE num_neurons;

                // Number of total active synapses.
                BGSIZE num_synapses;
        };

