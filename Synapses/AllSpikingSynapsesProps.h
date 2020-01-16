/**
 *      @file AllSpikingSynapsesProps.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "AllSynapsesProps.h"

/**
 * cereal
 */
//#include <cereal/types/polymorphic.hpp> //for inheritance
//#include <cereal/types/base_class.hpp> //for inherit parent's data member
//#include <cereal/types/vector.hpp>
//#include <vector>

class AllSpikingSynapsesProps : public AllSynapsesProps
{
    public:
        AllSpikingSynapsesProps();
        virtual ~AllSpikingSynapsesProps();

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapsesProps(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info);
        
        //! Cereal
        //template<class Archive>
        //void serialize(Archive & archive);

        /*template<class Archive>
        void save(Archive & archive) const;

        template<class Archive>
        void load(Archive & archive);*/

        /**
         *  Prints all SynapsesProps data.
         */
        virtual void printSynapsesProps();

#if defined(USE_GPU)
    public:
        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        virtual void setupSynapsesDeviceProps( void** allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesDeviceProps  Reference to the AllSpikingSynapsesProps class on device memory.
         */
        virtual void cleanupSynapsesDeviceProps( void* allSynapsesDeviceProps );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        virtual void copySynapseHostToDeviceProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        virtual void copySynapseDeviceToHostProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Get synapse_counts in AllSpikingSynapsesProps class on device memory.
         *
         *  @param  allSynapsesDeviceProps  Reference to the AllSpikingSynapsesProps class on device memory.
         *  @param  clr_info                ClusterInfo to refer from.
         */
        void copyDeviceSynapseCountsToHost(void* allSynapsesDeviceProps, const ClusterInfo *clr_info);

        /**
         *  Get sourceNeuronLayoutIndex and in_use in AllSpikingSynapsesProps class on device memory.
         *
         *  @param  allSynapsesDeviceProps  Reference to the AllSpikingSynapsesProps class on device memory.
         *  @param  sim_info                SimulationInfo to refer from.
         *  @param  clr_info                ClusterInfo to refer from.
         */
        void copyDeviceSourceNeuronIdxToHost(void* allSynapsesDeviceProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info);

        /**
         * Process inter clusters outgoing spikes.
         *
         *  @param  allSynapsesProps     Reference to the AllSpikingSynapsesProps struct
         *                                on device memory.
         */
        virtual void processInterClustesOutgoingSpikes(void* allSynapsesProps);

        /**
         * Process inter clusters incoming spikes.
         *
         *  @param  allSynapsesProps     Reference to the AllSpikingSynapsesProps struct
         *                                on device memory.
         */
        virtual void processInterClustesIncomingSpikes(void* allSynapsesProps);

        /**
         *  Prints all GPU SynapsesProps data.
         */
        virtual void printGPUSynapsesProps(void* allSynapsesDeviceProps );

    protected:
        /**
         *  Allocate GPU memories to store all synapses' states.
         *
         *  @param  allSynapsesProps      Reference to the AllSpikingSynapsesProps class.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void allocSynapsesDeviceProps( AllSpikingSynapsesProps &allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron);

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesProps  Reference to the AllSpikingSynapsesProps class.
         */
        void deleteSynapsesDeviceProps( AllSpikingSynapsesProps& allSynapsesProps );

        /**
         *  Copy all synapses' data from host to device.
         *  (Helper function of copySynapseHostToDeviceProps)
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllSpikingSynapsesProps class on device memory.
         *  @param  allSynapsesProps         Reference to the AllSpikingSynapsesProps class.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        void copyHostToDeviceProps( void* allSynapsesDeviceProps, AllSpikingSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *  (Helper function of copySynapseDeviceToHostProps)
         *
         *  @param  allSynapsesProps         Reference to the AllSpikingSynapsesProps class.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        void copyDeviceToHostProps( AllSpikingSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron);

        /**
         *  Prints all GPU SynapsesProps data.
         * (Helper function of printGPUSynapsesProps)
         */
        void printGPUSynapsesPropsHelper( AllSpikingSynapsesProps& allSynapsesProps);
#endif // USE_GPU

    public:
        /**
         *  Sets the data for Synapse to input's data.
         *
         *  @param  input  istream to read from.
         *  @param  iSyn   Index of the synapse to set.
         */
        virtual void readSynapseProps(istream &input, const BGSIZE iSyn);

        /**
         *  Write the synapse data to the stream.
         *
         *  @param  output  stream to print out to.
         *  @param  iSyn    Index of the synapse to print out.
         */
        virtual void writeSynapseProps(ostream& output, const BGSIZE iSyn) const;

    private:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        void cleanupSynapsesProps();

    public:
        /**
         *  The decay for the psr.
         */
        BGFLOAT *decay;

        /**
         *  The synaptic time constant \f$\tau\f$ [units=sec; range=(0,100)].
         */
        BGFLOAT *tau;

        /**
         *  The synaptic transmission delay, descretized into time steps.
         */
        int *total_delay;

        /**
         * The collection of synaptic transmission delay queue.
         */
        EventQueue *preSpikeQueue;
};

//! Cereal Serialization/Deserialization Method
/*template<class Archive>
void AllSpikingSynapsesProps::serialize(Archive & archive) {
    archive(cereal::base_class<AllSynapsesProps>(this));
}*/

/*template<class Archive>
void AllSpikingSynapsesProps::save(Archive & archive) const
{
    vector<BGFLOAT> decayVector;
    vector<BGFLOAT> tauVector;
    vector<int> total_delayVector;

    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        decayVector.push_back(decay[i]);
        tauVector.push_back(tau[i]);
        total_delayVector.push_back(total_delay[i]);
    }

    archive(cereal::base_class<AllSynapsesProps>(this),
    decayVector, tauVector, total_delayVector, *preSpikeQueue
    );
}

template<class Archive>
void AllSpikingSynapsesProps::load(Archive & archive) 
{
    vector<BGFLOAT> decayVector;
    vector<BGFLOAT> tauVector;
    vector<int> total_delayVector;

    archive(cereal::base_class<AllSynapsesProps>(this),
    decayVector, tauVector, total_delayVector, *preSpikeQueue
    );

    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        decay[i] = decayVector[i];
        tau[i] = tauVector[i];
        total_delay[i] = total_delayVector[i];
    }
}*/

//! Cereal
//CEREAL_REGISTER_TYPE(AllSpikingSynapsesProps)
