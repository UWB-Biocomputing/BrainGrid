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
#include "AllSpikingNeuronsProps.h"

class AllSpikingNeurons : public AllNeurons
{
    public:
        CUDA_CALLABLE AllSpikingNeurons();
        CUDA_CALLABLE virtual ~AllSpikingNeurons();

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
         *  @param  neuronsDevice          Pointer to the Neurons object in device memory.
         */
        virtual void advanceNeurons(IAllSynapses &synapses, void* allNeuronsDevice, void* allSynapsesDevice, const SimulationInfo *sim_info, float* randNoise, SynapseIndexMap* synapseIndexMapDevice, const ClusterInfo *clr_info, int iStepOffset, IAllNeurons* neuronsDevice);

        /**
         *  Set some parameters used for advanceNeuronsDevice.
         *
         *  @param  synapses               Reference to the allSynapses struct on host memory.
         */
        virtual void setAdvanceNeuronsDeviceParams(IAllSynapses &synapses);

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

#endif // !defined(USE_GPU)

#if defined(USE_GPU)

    public:
        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index                 Index of the Neuron to update.
         *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
         *  @param  deltaT                Inner simulation step duration.
         *  @param  simulationStep        The current simulation step.
         *  @param  pINeuronsProps        Pointer to the neurons properties.
         *  @param  randNoise             Pointer to device random noise array.
         */
        CUDA_CALLABLE virtual void advanceNeuron(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps, float* randNoise) = 0;

#else  // defined(USE_GPU)

    protected:
        /**
         *  Helper for #advanceNeuron. Updates state of a single neuron.
         *
         *  @param  index                 Index of the Neuron to update.
         *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
         *  @param  deltaT                Inner simulation step duration.
         *  @param  simulationStep        The current simulation step.
         *  @param  pINeuronsProps        Pointer to the neurons properties.
         *  @param  normRand              Pointer to the normalized random number generator.
         */
        virtual void advanceNeuron(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps, Norm* normRand) = 0;

#endif // defined(USE_GPU)

    protected:
        /**
         *  Initiates a firing of a neuron to connected neurons.
         *
         *  @param  index                 Index of the neuron to fire.
         *  @param  maxSpikes             Maximum number of spikes per neuron per epoch.
         *  @param  deltaT                Inner simulation step duration.
         *  @param  simulationStep        The current simulation step.
         *  @param  pINeuronsProps        Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void fire(const int index, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, IAllNeuronsProps* pINeuronsProps) const;

    protected:
        /**
         *  True if back propagaion is allowed.
         *  (parameters used for advanceNeuronsDevice.)
         */
        bool m_fAllowBackPropagation;

};

#if defined(USE_GPU)

/**
 *  CUDA code for advancing LIF neurons
 *
 *  @param[in] totalNeurons          Number of neurons.
 *  @param[in] maxSynapses           Maximum number of synapses per neuron.
 *  @param[in] maxSpikes             Maximum number of spikes per neuron per epoch.
 *  @param[in] deltaT                Inner simulation step duration.
 *  @param[in] simulationStep        The current simulation step.
 *  @param[in] randNoise             Pointer to device random noise array.
 *  @param[in] pINeuronsProps        Pointer to Neuron structures in device memory.
 *  @param[in] allSynapsesProps      Pointer to Synapse structures in device memory.
 *  @param[in] synapseIndexMap       Inverse map, which is a table indexed by an input neuron and maps to the synapses that provide input to that neuron.
 *  @param[in] fAllowBackPropagation True if back propagaion is allowed.
 *  @param[in] iStepOffset           Offset from the current simulation step.
 *  @param[in] neuronsDevice         Pointer to the Neurons object in device memory.
 */
extern __global__ void advanceNeuronsDevice( int totalNeurons, int maxSynapses, int maxSpikes, const BGFLOAT deltaT, uint64_t simulationStep, float* randNoise, IAllNeuronsProps* pINeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, SynapseIndexMap* synapseIndexMapDevice, bool fAllowBackPropagation, int iStepOffset, IAllNeurons* neuronsDevice );

#endif // USE_GPU
