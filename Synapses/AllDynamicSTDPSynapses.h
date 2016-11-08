/**
 *      @file AllDynamicSTDPSynapses.h
 *
 *      @brief A container of all dynamic STDP synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllDynamicSTDPSynapses AllDynamicSTDPSynapses.h "AllDynamicSTDPSynapses.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 *  The container holds synapse parameters of all synapses. 
 *  Each kind of synapse parameter is stored in a 2D array. Each item in the first 
 *  dimention of the array corresponds with each neuron, and each item in the second
 *  dimension of the array corresponds with a synapse parameter of each synapse of the neuron. 
 *  Bacause each neuron owns different number of synapses, the number of synapses 
 *  for each neuron is stored in a 1D array, synapse_counts.
 *
 *  For CUDA implementation, we used another structure, AllDSSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllDSSynapsesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion
 *  when copying data between host and device memory.
 *
 *  The AllDynamicSTDPSynapses inherited properties from the AllDSSynapses and the AllSTDPSynapses
 *  classes (multiple inheritance), and both the AllDSSynapses and the AllSTDPSynapses classes are
 *  the subclass of the AllSpikingSynapses class. Therefore, this is known as a diamond class
 *  inheritance, which causes the problem of ambibuous hierarchy compositon. To solve the
 *  problem, we can use the virtual inheritance. 
 *  However, the virtual inheritance will bring another problem. That is, we cannot static cast
 *  from a pointer to the AllSynapses class to a pointer to the AllDSSynapses or the AllSTDPSynapses 
 *  classes. Compiler requires dynamic casting because vtable mechanism is involed in solving the 
 *  casting. But the dynamic casting cannot be used for the pointers to device (CUDA) memories. 
 *  Considering these issues, I decided that making the AllDynamicSTDPSynapses class the subclass
 *  of the AllSTDPSynapses class and adding properties of the AllDSSynapses class to it (fumik).
 *   
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */
#pragma once

#include "AllSTDPSynapses.h"

struct AllDynamicSTDPSynapsesDeviceProperties;

class AllDynamicSTDPSynapses : public AllSTDPSynapses
{
    public:
        AllDynamicSTDPSynapses();
        AllDynamicSTDPSynapses(const int num_neurons, const int max_synapses);
        virtual ~AllDynamicSTDPSynapses();

        static IAllSynapses* Create() { return new AllDynamicSTDPSynapses(); }
 
        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         */
        virtual void setupSynapses(SimulationInfo *sim_info);

        /**
         *  Cleanup the class (deallocate memories).
         */
        virtual void cleanupSynapses();

        /**
         *  Reset time varying state vars and recompute decay.
         *
         *  @param  iSyn     Index of the synapse to set.
         *  @param  deltaT   Inner simulation step duration
         */
        virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);

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

        /**
         *  Create a Synapse and connect it to the model.
         *
         *  @param  synapses    The synapse list to reference.
         *  @param  iSyn        Index of the synapse to set.
         *  @param  source      Coordinates of the source Neuron.
         *  @param  dest        Coordinates of the destination Neuron.
         *  @param  sum_point   Summation point address.
         *  @param  deltaT      Inner simulation step duration.
         *  @param  type        Type of the Synapse to create.
         */
        virtual void createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type);

    protected:
        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         */
        virtual void setupSynapses(const int num_neurons, const int max_synapses);

        /**
         *  Sets the data for Synapse to input's data.
         *
         *  @param  input  istream to read from.
         *  @param  iSyn   Index of the synapse to set.
         */
        virtual void readSynapse(istream &input, const BGSIZE iSyn);

        /**
         *  Write the synapse data to the stream.
         *
         *  @param  output  stream to print out to.
         *  @param  iSyn    Index of the synapse to print out.
         */
        virtual void writeSynapse(ostream& output, const BGSIZE iSyn) const;

#if defined(USE_GPU)
    public:
        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, const SimulationInfo *sim_info );

        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allSynapsesDevice     Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        virtual void allocSynapseDeviceStruct( void** allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void deleteSynapseDeviceStruct( void* allSynapsesDevice );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, const SimulationInfo *sim_info );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        virtual void copySynapseHostToDevice( void* allSynapsesDevice, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void copySynapseDeviceToHost( void* allSynapsesDevice, const SimulationInfo *sim_info );

        /**
         *  Set synapse class ID defined by enumClassSynapses for the caller's Synapse class.
         *  The class ID will be set to classSynapses_d in device memory,
         *  and the classSynapses_d will be referred to call a device function for the
         *  particular synapse class.         
         *  Because we cannot use virtual function (Polymorphism) in device functions,
         *  we use this scheme.         
         *  Note: we used to use a function pointer; however, it caused the growth_cuda crash
         *  (see issue#137).
         */
        virtual void setSynapseClassID();

    protected:
        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *  (Helper function of allocSynapseDeviceStruct)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void allocDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties &allSynapses, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Delete GPU memories.
         *  (Helper function of deleteSynapseDeviceStruct)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         */
        void deleteDeviceStruct( AllDynamicSTDPSynapsesDeviceProperties& allSynapses );

        /**
         *  Copy all synapses' data from host to device.
         *  (Helper function of copySynapseHostToDevice)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void copyHostToDevice( void* allSynapsesDevice, AllDynamicSTDPSynapsesDeviceProperties& allSynapses, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *  (Helper function of copySynapseDeviceToHost)
         *
         *  @param  allSynapsesDevice  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void copyDeviceToHost( AllDynamicSTDPSynapsesDeviceProperties& allSynapses, const SimulationInfo *sim_info );
#else // !defined(USE_GPU)
    protected:
        /**
         *  Calculate the post synapse response after a spike.
         *
         *  @param  iSyn        Index of the synapse to set.
         *  @param  deltaT      Inner simulation step duration.
         */
        virtual void changePSR(const BGSIZE iSyn, const BGFLOAT deltaT);
#endif // defined(USE_GPU)
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

#if defined(USE_GPU)
struct AllDynamicSTDPSynapsesDeviceProperties : public AllSTDPSynapsesDeviceProperties
{
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
#endif // defined(USE_GPU)

