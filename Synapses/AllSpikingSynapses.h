/**
 *      @file AllSpikingSynapses.h
 *
 *      @brief A container of all spiking synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSpikingSynapses AllSpikingSynapses.h "AllSpikingSynapses.h"
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
 *  For CUDA implementation, we used another structure, AllSynapsesDevice, where synapse
 *  parameters are stored in 1D arrays instead of 2D arrays, so that device functions
 *  can access these data less latency. When copying a synapse parameter, P[i][j],
 *  from host to device, it is stored in P[i * max_synapses_per_neuron + j] in 
 *  AllSynapsesDevice structure.
 *
 *  The latest implementation uses the identical data struture between host and CUDA;
 *  that is, synapse parameters are stored in a 1D array, so we don't need conversion
 *  when copying data between host and device memory.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */
#pragma once

#include "AllSynapses.h"
#include "EventQueue.h"
#include "AllSpikingSynapsesProperties.h"

typedef void (*fpPreSynapsesSpikeHit_t)(const BGSIZE, AllSpikingSynapsesProperties*);
typedef void (*fpPostSynapsesSpikeHit_t)(const BGSIZE, AllSpikingSynapsesProperties*);

#include "AllSpikingNeurons.h"

class AllSpikingSynapses : public AllSynapses
{
    public:
        AllSpikingSynapses();
        AllSpikingSynapses(const AllSpikingSynapses &r_synapses);
        virtual ~AllSpikingSynapses();

        static IAllSynapses* Create() { return new AllSpikingSynapses(); }

        /**
         *  Assignment operator: copy synapses parameters.
         *
         *  @param  r_synapses  Synapses class object to copy from.
         */
        virtual IAllSynapses &operator=(const IAllSynapses &r_synapses);

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapses(SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info      SimulationInfo class to read information from.
         *  @param  clr_info      ClusterInfo class to read information from.
         */
        virtual void setupSynapses(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info);

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
         *  Sets the data for Synapses to input's data.
         *
         *  @param  input  istream to read from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clrm_info);

        /**
         *  Write the synapses data to the stream.
         *
         *  @param  output  stream to print out to.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void serialize(ostream& output, const ClusterInfo *clr_info);

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

        /**
         *  Check if the back propagation (notify a spike event to the pre neuron)
         *  is allowed in the synapse class.
         *
         *  @retrun true if the back propagation is allowed.
         */
        virtual bool allowBackPropagation();

    protected:
        /**
         *  Setup the internal structure of the class.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        void setupSynapsesInternalState(SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Deallocate all resources.
         */
        void cleanupSynapsesInternalState();

        /**
         *  Copy synapses parameters.
         *
         *  @param  r_synapses  Synapses class object to copy from.
         */
        void copyParameters(const AllSpikingSynapses &r_synapses);

        /**
         *  Updates the decay if the synapse selected.
         *
         *  @param  iSyn    Index of the synapse to set.
         *  @param  deltaT  Inner simulation step duration
         *  @return true is success.
         */
        bool updateDecay(const BGSIZE iSyn, const BGFLOAT deltaT);

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
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void allocSynapseDeviceStruct( void** allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Allocate GPU memories to store all synapses' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allSynapsesProperties     Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         *  @param  clusterID             The cluster ID of the cluster.
         */
        virtual void allocSynapseDeviceStruct( void** allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID );

        /**
         *  Delete GPU memories.
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         */
        virtual void deleteSynapseDeviceStruct( void* allSynapsesProperties );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copySynapseHostToDevice( void* allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy all synapses' data from host to device.
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        virtual void copySynapseHostToDevice( void* allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron );
        /**
         *  Copy all synapses' data from device to host.
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copySynapseDeviceToHost( void* allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Get synapse_counts in AllSynapses struct on device memory.
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyDeviceSynapseCountsToHost(void* allSynapsesProperties, const ClusterInfo *clr_info);

        /** 
         *  Get sourceNeuronLayoutIndex and in_use in AllSynapses struct on device memory.
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyDeviceSourceNeuronIdxToHost(void* allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info);

        /**
         *  Advance all the Synapses in the simulation.
         *  Update the state of all synapses for a time step.
         *
         *  @param  allSynapsesProperties      Reference to the allSynapses struct on device memory.
         *  @param  allNeuronsProperties       Reference to the allNeurons struct on device memory.
         *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
         *  @param  sim_info               SimulationInfo class to read information from.
         *  @param  clr_info               ClusterInfo to refer from.
         *  @param  iStepOffset            Offset from the current simulation step.
         */
        virtual void advanceSynapses(void* allSynapsesProperties, void* allNeuronsProperties, void* synapseIndexMapDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset);

        /**
         * Process inter clusters outgoing spikes.
         *
         *  @param  allSynapsesProperties     Reference to the AllSpikingSynapsesProperties struct
         *                                on device memory.
         */
        virtual void processInterClustesOutgoingSpikes(void* allSynapsesProperties);

        /**
         * Process inter clusters incoming spikes.
         *
         *  @param  allSynapsesProperties     Reference to the AllSpikingSynapsesProperties struct
         *                                on device memory.
         */
        virtual void processInterClustesIncomingSpikes(void* allSynapsesProperties);

        /**
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         *  @param  allSynapsesProperties      Reference to the AllSynapsesProperties struct
         *                                 on device memory.
         *  @param  iStep                  Simulation steps to advance.
         */
        virtual void advanceSpikeQueue(void* allSynapsesProperties, int iStep);

        /**
         *  Set some parameters used for advanceSynapsesDevice.
         *  Currently we set a member variable: m_fpChangePSR_h.
         */
        virtual void setAdvanceSynapsesDeviceParams();

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
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         *  @param  clusterID             The cluster ID of the cluster.
         */
        void allocDeviceStruct( AllSpikingSynapsesProperties &allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron, CLUSTER_INDEX_TYPE clusterID );

        /**
         *  Delete GPU memories.
         *  (Helper function of deleteSynapseDeviceStruct)
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         */
        void deleteDeviceStruct( AllSpikingSynapsesProperties& allSynapsesPorperties );

        /**
         *  Copy all synapses' data from host to device.
         *  (Helper function of copySynapseHostToDevice)
         *
         *  @param  allSynapsesDeviceProperties  Reference to the allSynapses struct on device memory.
         *  @param  num_neurons           Number of neurons.
         *  @param  maxSynapsesPerNeuron  Maximum number of synapses per neuron.
         */
        void copyHostToDevice( void* allSynapsesDeviceProperties, AllSpikingSynapsesProperties& allSynapsesProperties, int num_neurons, int maxSynapsesPerNeuron );

        /**
         *  Copy all synapses' data from device to host.
         *  (Helper function of copySynapseDeviceToHost)
         *
         *  @param  allSynapsesProperties  Reference to the allSynapses struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        void copyDeviceToHost( AllSpikingSynapsesProperties& allSynapsesProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info );
#else  // !defined(USE_GPU)
public:
        /**
         *  Advance all the Synapses in the simulation.
         *  Update the state of all synapses for a time step.
         *
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  neurons          The Neuron list to search from.
         *  @param  synapseIndexMap  Pointer to the synapse index map.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void advanceSynapses(const SimulationInfo *sim_info, IAllNeurons *neurons, SynapseIndexMap *synapseIndexMap, int iStepOffset);

        /*
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         * @param iStep     simulation steps to advance.
         */
        virtual void advanceSpikeQueue(int iStep);

        /**
         *  Advance one specific Synapse.
         *
         *  @param  iSyn      Index of the Synapse to connect to.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  neurons   The Neuron list to search from.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void advanceSynapse(const BGSIZE iSyn, const SimulationInfo *sim_info, IAllNeurons *neurons, int iStepOffset);

        /**
         *  Prepares Synapse for a spike hit.
         *
         *  @param  iSyn      Index of the Synapse to update.
         *  @param  iCluster  Cluster ID of cluster where the spike is added.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void preSpikeHit(const BGSIZE iSyn, const CLUSTER_INDEX_TYPE iCluster, int iStepOffset);

        /**
         *  Prepares Synapse for a spike hit (for back propagation).
         *
         *  @param  iSyn   Index of the Synapse to update.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void postSpikeHit(const BGSIZE iSyn, int iStepOffset);

    protected:
        /**
         *  Checks if there is an input spike in the queue.
         *
         *  @param  iSyn   Index of the Synapse to connect to.
         *  @param  iStepOffset      Offset from the current simulation step.
         *  @return true if there is an input spike event.
         */
        bool isSpikeQueue(const BGSIZE iSyn, int iStepOffset);

        /**
         *  Calculate the post synapse response after a spike.
         *
         *  @param  iSyn        Index of the synapse to set.
         *  @param  deltaT      Inner simulation step duration.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, int iStepOffset);
#endif
};

