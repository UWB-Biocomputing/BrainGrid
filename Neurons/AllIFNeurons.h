/**
 *      @file AllIFNeurons.h
 *
 *      @brief A container of all Integate and Fire (IF) neuron data
 */

/** 
 ** @authors Aaron Oziel, Sean Blackbourn
 **
 ** @class AllIFNeurons AllIFNeurons.h "AllIFNeurons.h"
 **
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 ** A container of all Integate and Fire (IF) neuron data.
 ** This is the base class of all Integate and Fire (IF) neuron classes.
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

#include "Global.h"
#include "AllSpikingNeurons.h"
#include "AllIFNeuronsProps.h"

class AllIFNeurons : public AllSpikingNeurons
{
    public:
        AllIFNeurons();
        virtual ~AllIFNeurons();

        /**
         *  Create and setup neurons properties.
         */
        virtual void createNeuronsProps();

#if defined(USE_GPU)
    public:
        /**
         *  Update the state of all neurons for a time step
         *  Notify outgoing synapses if neuron has fired.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         *  @param  allNeuronsDevice       Reference to the allNeurons struct on device memory.
         *  @param  allSynapsesDevice      Reference to the allSynapses struct on device memory.
         *  @param  sim_info               SimulationInfo to refer from.
         *  @param  randNoise              Reference to the random noise array.
         *  @param  synapseIndexMapDevice  Reference to the SynapseIndexMap on device memory.
         *  @param  clr_info               ClusterInfo to refer from.
         *  @param  iStepOffset            Offset from the current simulation step.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset);

        /**
         *  Allocate GPU memories to store all neurons' states,
         *  and copy them from host to GPU memory.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void allocNeuronDeviceStruct( void** allNeuronsDevice, SimulationInfo *sim_info, ClusterInfo *clr_info );

        /**
         *  Delete GPU memories.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void deleteNeuronDeviceStruct( void* allNeuronsDevice, const ClusterInfo *clr_info );

        /**
         *  Copy all neurons' data from host to device.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyNeuronHostToDevice( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy all neurons' data from device to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyNeuronDeviceToHost( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy spike history data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeHistoryToHost( void* allNeuronsDevice, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy spike counts data stored in device memory to host.
         *
         *  @param  allNeuronsDevice   Reference to the allNeurons struct on device memory.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        virtual void copyNeuronDeviceSpikeCountsToHost( void* allNeuronsDevice, const ClusterInfo *clr_info );

    protected:
        /**
         *  Allocate GPU memories to store all neurons' states.
         *  (Helper function of allocNeuronDeviceStruct)
         *
         *  @param  allNeuronsProps         Reference to the AllIFNeuronsProps struct.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        void allocDeviceStruct( AllIFNeuronsProps &allNeuronsProps, SimulationInfo *sim_info, ClusterInfo *clr_info );

        /**
         *  Delete GPU memories.
         *  (Helper function of deleteNeuronDeviceStruct)
         *
         *  @param  allNeuronsProps         Reference to the AllIFNeuronsProps struct.
         *  @param  clr_info           ClusterInfo to refer from.
         */
        void deleteDeviceStruct( AllIFNeuronsProps& allNeuronsProps, const ClusterInfo *clr_info );

        /**
         *  Copy all neurons' data from host to device.
         *  (Helper function of copyNeuronHostToDevice)
         *
         *  @param  allNeuronsProps         Reference to the AllIFNeuronsProps struct.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
	void copyHostToDevice( AllIFNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

        /**
         *  Copy all neurons' data from device to host.
         *  (Helper function of copyNeuronDeviceToHost)
         *
         *  @param  allNeuronsProps         Reference to the AllIFNeuronsProps struct.
         *  @param  sim_info           SimulationInfo to refer from.
         *  @param  clr_info           ClusterInfo to refer from.
         */
	void copyDeviceToHost( AllIFNeuronsProps& allNeuronsProps, const SimulationInfo *sim_info, const ClusterInfo *clr_info );

#endif // defined(USE_GPU)
};

