#pragma once
        
        struct SynapseIndexMap
        {
                //! The beginning index of the incoming dynamic spiking synapse array.
                int* incomingSynapse_begin;

                //! The number of incoming synapses.
                int* synapseCount;

                //! Pointer to the synapse inverse map.
                uint32_t* inverseIndex;

                //! Pointer to the active synapse map.
                uint32_t* activeSynapseIndex;

                SynapseIndexMap() : num_neurons(0), num_synapses(0)
                {
                        incomingSynapse_begin = NULL;
                        synapseCount = NULL;
                        inverseIndex = NULL;
                        activeSynapseIndex = NULL;
                };

                SynapseIndexMap(int neuron_count, int synapse_count) : num_neurons(neuron_count), num_synapses(synapse_count)
                {
                        incomingSynapse_begin = new int[neuron_count];
                        synapseCount = new int[neuron_count];
                        inverseIndex = new uint32_t[synapse_count];
                        activeSynapseIndex = new uint32_t[synapse_count];
                };

                ~SynapseIndexMap()
                {
                        if (num_neurons != 0) {
                                delete[] incomingSynapse_begin;
                                delete[] synapseCount;
                        }
                        if (num_synapses != 0) {
                                delete[] inverseIndex;
                                delete[] activeSynapseIndex;
                        }
                }

        private:
                int num_neurons;
                int num_synapses;
        };

