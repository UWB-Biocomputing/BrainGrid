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
#include "AllSpikingSynapsesProps.h"

typedef void (*fpPreSynapsesSpikeHit_t)(const BGSIZE, AllSpikingSynapsesProps*);
typedef void (*fpPostSynapsesSpikeHit_t)(const BGSIZE, AllSpikingSynapsesProps*);

#include "AllSpikingNeurons.h"

class AllSpikingSynapses : public AllSynapses
{
    public:
        CUDA_CALLABLE AllSpikingSynapses();
        CUDA_CALLABLE virtual ~AllSpikingSynapses();

        static IAllSynapses* Create() { return new AllSpikingSynapses(); }

        /**
         *  Create and setup synapses properties.
         */
        virtual void createSynapsesProps();

        /**
         *  Reset time varying state vars and recompute decay.
         *
         *  @param  iSyn     Index of the synapse to set.
         *  @param  deltaT   Inner simulation step duration
         */
        virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);

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
         *  Updates the decay if the synapse selected.
         *
         *  @param  iSyn    Index of the synapse to set.
         *  @param  deltaT  Inner simulation step duration
         *  @return true is success.
         */
        bool updateDecay(const BGSIZE iSyn, const BGFLOAT deltaT);

#if defined(USE_GPU)

    public:
        /**
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         *  @param  allSynapsesProps      Reference to the AllSynapsesProps struct
         *                                 on device memory.
         *  @param  iStep                  Simulation steps to advance.
         */
        virtual void advanceSpikeQueue(void* allSynapsesProps, int iStep);

        /**
         *  Create a AllSynapses class object in device
         *
         *  @param pAllSynapses_d      Device memory address to save the pointer of created AllSynapses object.
         *  @param pAllSynapsesProps_d  Pointer to the synapses properties in device memory.
         */
        virtual void createAllSynapsesInDevice(IAllSynapses** pAllSynapses_d, IAllSynapsesProps *pAllSynapsesProps_d);

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
#endif  // defined(USE_GPU)

public:
        /**
         *  Advance one specific Synapse.
         *
         *  @param  iSyn             Index of the Synapse to connect to.
         *  @param  deltaT           Inner simulation step duration.
         *  @param  neurons          The Neuron list to search from.
         *  @param  simulationStep   The current simulation step.
         *  @param  iStepOffset      Offset from the current global simulation step.
         *  @param  maxSpikes        Maximum number of spikes per neuron per epoch.
         *  @param  pISynapsesProps  Pointer to the synapses properties.
         *  @param  pINeuronsProps   Pointer to the neurons properties.
         */
        CUDA_CALLABLE virtual void advanceSynapse(const BGSIZE iSyn, const BGFLOAT deltaT, IAllNeurons *neurons, uint64_t simulationStep, int iStepOffset, int maxSpikes, IAllSynapsesProps* pISynapsesProps, IAllNeuronsProps* pINeuronsProps);

#if !defined(USE_GPU)
        /**
         *  Prepares Synapse for a spike hit.
         *
         *  @param  iSyn      Index of the Synapse to update.
         *  @param  iCluster  Cluster ID of cluster where the spike is added.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        CUDA_CALLABLE virtual void preSpikeHit(const BGSIZE iSyn, const CLUSTER_INDEX_TYPE iCluster, int iStepOffset);

        /**
         *  Prepares Synapse for a spike hit (for back propagation).
         *
         *  @param  iSyn   Index of the Synapse to update.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        CUDA_CALLABLE virtual void postSpikeHit(const BGSIZE iSyn, int iStepOffset);

        /*
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         * @param iStep     simulation steps to advance.
         */
        CUDA_CALLABLE virtual void advanceSpikeQueue(int iStep);
#endif

    protected:
        /**
         *  Checks if there is an input spike in the queue.
         *
         *  @param  iSyn             Index of the Synapse to connect to.
         *  @param  iStepOffset      Offset from the current global simulation step.
         *  @param  pISynapsesProps  Pointer to the synapses properties.
         *  @return true if there is an input spike event.
         */
        CUDA_CALLABLE bool isSpikeQueue(const BGSIZE iSyn, int iStepOffset, IAllSynapsesProps* pISynapsesProps);

        /**
         *  Calculate the post synapse response after a spike.
         *
         *  @param  iSyn             Index of the synapse to set.
         *  @param  deltaT           Inner simulation step duration.
         *  @param  simulationStep   The current simulation step.
         *  @param  pISynapsesProps  Pointer to the synapses properties.
         */
        CUDA_CALLABLE virtual void changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, uint64_t simulationStep, IAllSynapsesProps* pISynapsesProps);
};

#if defined(USE_GPU)

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllSpikingSynapsesDevice(IAllSynapses **pAllSynapses, IAllSynapsesProps *pAllSynapsesProps);

#endif // USE_GPU
