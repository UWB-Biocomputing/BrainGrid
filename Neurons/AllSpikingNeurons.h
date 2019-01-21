/**
 *      @file AllSpikingNeurons.h
 *
 *      @brief A container of all spiking neuron data
 */

/**
 ** @class AllSpikingNeurons AllSpikingNeurons.h "AllSpikingNeurons.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** A container of all spiking neuron data.
 ** This is the base class of all spiking neuron classes.
 **
 ** The class uses a data-centric structure, which utilizes a structure as the containers of
 ** all neuron.
 **
 ** The container holds neuron parameters of all neurons.
 ** Each kind of neuron parameter is stored in a 1D array, of which length
 ** is number of all neurons. Each array of a neuron parameter is pointed by a
 ** corresponding member variable of the neuron parameter in the class.
 ** 
 ** This structure was originally designed for the GPU implementation of the
 ** simulator, and this refactored version of the simulator simply uses that design for
 ** all other implementations as well. This is to simplify transitioning from
 ** single-threaded to multi-threaded.
 **
 ** \latexonly  \subsubsection*{Credits} \endlatexonly
 ** \htmlonly   <h3>Credits</h3> \endhtmlonly
 **
 ** Some models in this simulator is a rewrite of CSIM (2006) and other
 ** work (Stiber and Kawasaki (2007?))
 **
 **/

#pragma once

using namespace std;

#include "Global.h"
#include "SimulationInfo.h"
#include "AllNeurons.h"
#include "AllSpikingSynapses.h"
#include "AllSpikingNeuronsProperties.h"

class AllSpikingNeurons : public AllNeurons
{
    public:
        AllSpikingNeurons();
        virtual ~AllSpikingNeurons();

        /**
         *  Clear the spike counts out of all Neurons.
         *
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         *  @param  clr       Cluster class to read information from.
         */
        void clearSpikeCounts(const SimulationInfo *sim_info, const ClusterInfo *clr_info, Cluster *clr);

#if defined(USE_GPU)
    public:
        /**
         *  Set some parameters used for advanceNeuronsDevice.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         */
        virtual void setAdvanceNeuronsDeviceParams(IAllSynapses &synapses);

        /**
         *  Copy spike counts data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const ClusterInfo *clr_info ) = 0;

        /**
         *  Copy spike history data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info ) = 0;

    protected:
        /**
         *  Copy spike history data stored in device memory to host.
         *  (Helper function of copyNeuronDeviceSpikeHistoryToHost)
         *
         *  @param  allNeuronsProperties        Reference to the AllSpikingNeuronsProperties struct.
         *  @param  sim_info          SimulationInfo to refer from.
         *  @param  clr_info          ClusterInfo to refer from.
         */
        void copyDeviceSpikeHistoryToHost( AllSpikingNeuronsProperties& allNeuronsProperties, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy spike counts data stored in device memory to host.
         *  (Helper function of copyNeuronDeviceSpikeCountsToHost)
         *
         *  @param  allNeuronsProperties         Reference to the AllSpikingNeuronsProperties struct.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        void copyDeviceSpikeCountsToHost( AllSpikingNeuronsProperties& allNeuronsProperties, const ClusterInfo *clr_info );

#else // !defined(USE_GPU)

    public:
        /**
         *  Update internal state of the indexed Neuron (called by every simulation step).
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses         The Synapse list to search from.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  synapseIndexMap  Reference to the SynapseIndexMap.
         *  @param  clr_info         ClusterInfo class to read information from.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, const SimulationInfo *sim_info, const SynapseIndexMap *synapseIndexMap, const ClusterInfo *clr_info, int iStepOffset);

        /**
         *  Get the spike history of neuron[index] at the location offIndex.
         *
         *  @param  index            Index of the neuron to get spike history.
         *  @param  offIndex         Offset of the history buffer to get from.
         *  @param  sim_info         SimulationInfo class to read information from.
         */
        uint64_t getSpikeHistory(int index, int offIndex, const SimulationInfo *sim_info);

    protected:
        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index            Index of the neuron to update.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  clr_info         ClusterInfo class to read information from.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void advanceNeuron(const int index, const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStepOffset) = 0;

        /**
         *  Initiates a firing of a neuron to connected neurons
         *
         *  @param  index            Index of the neuron to fire.
         *  @param  sim_info         SimulationInfo class to read information from.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void fire(const int index, const SimulationInfo *sim_info, int iStepOffset) const;
#endif // defined(USE_GPU)

    protected:
        /**
         *  True if back propagaion is allowed.
         *  (parameters used for advanceNeuronsDevice.)
         */
        bool m_fAllowBackPropagation;

};

