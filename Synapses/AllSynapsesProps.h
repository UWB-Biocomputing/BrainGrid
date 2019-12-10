/**
 *      @file AllSynapsesProps.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "IAllSynapsesProps.h"

/**
 * cereal
 */
#include <cereal/types/polymorphic.hpp> //for inheritance
#include <cereal/types/vector.hpp>
#include <vector>

class AllSynapsesProps : public IAllSynapsesProps
{
    public:
        AllSynapsesProps();
        virtual ~AllSynapsesProps();

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
        template<class Archive>
        void serialize(Archive & archive);

#if defined(USE_GPU)
    protected:
        /**
         *  Allocate GPU memories to store all synapses' states.
         *
         *  @param  allSynapsesProps      Reference to the AllSynapsesProps class.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void allocSynapsesDeviceProps( AllSynapsesProps &allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron);

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesProps  Reference to the AllSynapsesProps class.
         */
        void deleteSynapsesDeviceProps( AllSynapsesProps& allSynapsesProps );

        /**
         *  Copy all synapses' data from host to device.
         *  (Helper function of copySynapseHostToDeviceProps)
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllSynapsesProps class on device memory.
         *  @param  allSynapsesProps         Reference to the AllSynapsesProps class.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        void copyHostToDeviceProps( void* allSynapsesDeviceProps, AllSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *  (Helper function of copySynapseDeviceToHostProps)
         *
         *  @param  allSynapsesProps         Reference to the AllSynapsesProps class.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        void copyDeviceToHostProps( AllSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron);
#endif // USE_GPU

    public:
        /**
         *  Checks the number of required parameters to read.
         *
         * @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters();

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  element TiXmlElement to examine.
         *  @return true if successful, false otherwise.
         */
        virtual bool readParameters(const TiXmlElement& element);

        /**
         *  Prints out all parameters of the neurons to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const;

        /*
         *  Copy synapses parameters.
         *
         *  @param  r_synapsesProps  Synapses properties class object to copy from.
         */
        virtual void copyParameters(const AllSynapsesProps *r_synapsesProps);

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

        /**
         *  Returns an appropriate synapseType object for the given integer.
         *
         *  @param  type_ordinal    Integer that correspond with a synapseType.
         *  @return the SynapseType that corresponds with the given integer.
         */
        synapseType synapseOrdinalToType(const int type_ordinal);

    protected:
        /**
         *  Number of parameters read.
         */
        int nParams;

    public:
        /**
         *  The location of the source neuron
         */
        int *sourceNeuronLayoutIndex;

        /**
         *  The location of the destination neuron
         */
        int *destNeuronLayoutIndex;

        /**
         *   The weight (scaling factor, strength, maximal amplitude) of the synapse.
         */
         BGFLOAT *W;

        /**
         *  This synapse's summation point's address.
         */
        BGFLOAT **summationPoint;

        /**
         *  Synapse type
         */
        synapseType *type;

        /**
         *  The post-synaptic response is the result of whatever computation
         *  is going on in the synapse.
         */
        BGFLOAT *psr;

        /**
         *  The boolean value indicating the entry in the array is in use.
         */
        bool *in_use;

        /**
         *  The number of synapses for each neuron.
         *  Note: Likely under a different name in GpuSim_struct, see synapse_count. -Aaron
         */
        BGSIZE *synapse_counts;

        /**
         *  The total number of active synapses.
         */
        BGSIZE total_synapse_counts;

        /**
         *  The maximum number of synapses for each neurons.
         */
        BGSIZE maxSynapsesPerNeuron;

        /**
         *  The number of neurons
         *  Aaron: Is this even supposed to be here?!
         *  Usage: Used by destructor
         */
        int count_neurons;

        /**
         *  A temporary variable used for parallel reduction in calcSummationMapDevice.
         */
        BGFLOAT *summation;
};

//! Cereal Serialization/Deserialization Method
template<class Archive>
void AllSynapsesProps::serialize(Archive & archive) {
    vector<BGFLOAT> temp;
    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        temp.push_back(W[i]);
    }
    archive(temp);
}

//! Cereal
CEREAL_REGISTER_TYPE(AllSynapsesProps)
CEREAL_REGISTER_POLYMORPHIC_RELATION(IAllSynapsesProps,AllSynapsesProps)
