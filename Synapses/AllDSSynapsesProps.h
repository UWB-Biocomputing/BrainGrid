/**
 *      @file AllDSSynapsesProps.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "AllSpikingSynapsesProps.h"

/**
 * cereal
 */
//#include <cereal/types/polymorphic.hpp> //for inheritance
//#include <cereal/types/base_class.hpp> //for inherit parent's data member
//#include <cereal/types/vector.hpp>
//#include <vector>

class AllDSSynapsesProps : public AllSpikingSynapsesProps
{
    public:
        AllDSSynapsesProps();
        virtual ~AllDSSynapsesProps();

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
         *  @param  allSynapsesDeviceProps   Reference to the AllDSSynapsesProps class on device memory.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        virtual void setupSynapsesDeviceProps( void** allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesDeviceProps  Reference to the AllDSSynapsesProps class on device memory.
         */
        virtual void cleanupSynapsesDeviceProps( void* allSynapsesDeviceProps );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllDSSynapsesProps class on device memory.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        virtual void copySynapseHostToDeviceProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllDSSynapsesProps class on device memory.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        virtual void copySynapseDeviceToHostProps( void* allSynapsesDeviceProps, int num_neurons, int maxSynapsesPerNeuron );
        
        /**
         *  Prints all GPU SynapsesProps data.
         */
        virtual void printGPUSynapsesProps(void* allSynapsesDeviceProps );
    protected:
        /**
         *  Allocate GPU memories to store all synapses' states.
         *
         *  @param  allSynapsesProps      Reference to the AllDSSynapsesProps class.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void allocSynapsesDeviceProps( AllDSSynapsesProps &allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron);

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesProps  Reference to the AllDSSynapsesProps class.
         */
        void deleteSynapsesDeviceProps( AllDSSynapsesProps& allSynapsesProps );

        /**
         *  Copy all synapses' data from host to device.
         *  (Helper function of copySynapseHostToDeviceProps)
         *
         *  @param  allSynapsesDeviceProps   Reference to the AllDSSynapsesProps class on device memory.
         *  @param  allSynapsesProps         Reference to the AllDSSynapsesProps class.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        void copyHostToDeviceProps( void* allSynapsesDeviceProps, AllDSSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *  (Helper function of copySynapseDeviceToHostProps)
         *
         *  @param  allSynapsesProps         Reference to the AllDSSynapsesProps class.
         *  @param  num_neurons              Number of neurons.
         *  @param  maxSynapsesPerNeuron     Maximum number of synapses per neuron.
         */
        void copyDeviceToHostProps( AllDSSynapsesProps& allSynapsesProps, int num_neurons, int maxSynapsesPerNeuron);
        
        /**
         *  Prints all GPU SynapsesProps data.
         * (Helper function of printGPUSynapsesProps)
         */
        void printGPUSynapsesPropsHelper( AllDSSynapsesProps& allSynapsesProps);
#endif // USE_GPU

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
         *  The time of the last spike.
         */
        uint64_t *lastSpike;

        /**
         *  The time varying state variable \f$r\f$ for depression.
         */
        BGFLOAT *r;

        /**
         *  The time varying state variable \f$u\f$ for facilitation.
         */
        BGFLOAT *u;

        /**
         *  The time constant of the depression of the dynamic synapse [range=(0,10); units=sec].
         */
        BGFLOAT *D;

        /**
         *  The use parameter of the dynamic synapse [range=(1e-5,1)].
         */
        BGFLOAT *U;

        /**
         *  The time constant of the facilitation of the dynamic synapse [range=(0,10); units=sec].
         */
        BGFLOAT *F;
};

//! Cereal Serialization/Deserialization Method
/*template<class Archive>
void AllDSSynapsesProps::serialize(Archive & archive) {
    archive(cereal::base_class<AllSpikingSynapsesProps>(this));
}*/

/*template<class Archive>
void AllDSSynapsesProps::save(Archive & archive) const
{
    vector<uint64_t> lastSpikeVector;
    vector<BGFLOAT> rVector;
    vector<BGFLOAT> uVector;
    vector<BGFLOAT> DVector;
    vector<BGFLOAT> UVector;
    vector<BGFLOAT> FVector;

    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        lastSpikeVector.push_back(lastSpike[i]);
        rVector.push_back(r[i]);
        uVector.push_back(u[i]);
        DVector.push_back(D[i]);
        UVector.push_back(U[i]);
        FVector.push_back(F[i]);

    }

    archive(cereal::base_class<AllSpikingSynapsesProps>(this),
    lastSpikeVector, rVector, uVector,
    DVector, UVector, FVector);
}

template<class Archive>
void AllDSSynapsesProps::load(Archive & archive) 
{
    vector<uint64_t> lastSpikeVector;
    vector<BGFLOAT> rVector;
    vector<BGFLOAT> uVector;
    vector<BGFLOAT> DVector;
    vector<BGFLOAT> UVector;
    vector<BGFLOAT> FVector;

    archive(cereal::base_class<AllSpikingSynapsesProps>(this),
    lastSpikeVector, rVector, uVector,
    DVector, UVector, FVector);

    for(int i = 0; i < maxSynapsesPerNeuron * count_neurons; i++) {
        lastSpike[i] = lastSpikeVector[i];
        r[i] = rVector[i];
        u[i] = uVector[i];
        D[i] = DVector[i];
        U[i] = UVector[i];
        F[i] = FVector[i];
    }
}*/

//! Cereal
//CEREAL_REGISTER_TYPE(AllDSSynapsesProps)
