/**
 *      @file AllSTDPSynapses.h
 *
 *      @brief A container of all STDP synapse data
 */

/** 
 * @authors Aaron Oziel, Sean Blackbourn
 * 
 * @class AllSTDPSynapses AllSTDPSynapses.h "AllSTDPSynapses.h"
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
 */

/** 
 *  Implements the basic weight update for a time difference \f$Delta =
 *  t_{post}-t_{pre}\f$ with presynaptic spike at time \f$t_{pre}\f$ and
 *  postsynaptic spike at time \f$t_{post}\f$. Then, the weight update is given by
 *  \f$dw =  Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$, and \f$dw =  Aneg *
 *  exp(-Delta/tauneg)\f$ for \f$Delta < 0\f$. (set \f$useFroemkeDanSTDP=0\f$ and
 *  \f$mupos=muneg=0\f$ for this basic update rule).
 *  
 *  It is also possible to use an
 *  extended multiplicative update by changing mupos and muneg. Then \f$dw =
 *  (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$ and \f$dw =
 *  W^{mupos} * Aneg * exp(Delta/tauneg)\f$ for \f$Delta < 0\f$. (see Guetig,
 *  Aharonov, Rotter and Sompolinsky (2003). Learning input correlations through
 *  non-linear asymmetric Hebbian plasticity. Journal of Neuroscience 23.
 *  pp.3697-3714.)
 *      
 *  Set \f$useFroemkeDanSTDP=1\f$ (this is the default value) and
 *  use \f$tauspost\f$ and \f$tauspre\f$ for the rule given in Froemke and Dan
 *  (2002). Spike-timing-dependent synaptic modification induced by natural spike
 *  trains. Nature 416 (3/2002). 
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "AllSpikingSynapses.h"
#include "AllSTDPSynapsesProps.h"

class AllSTDPSynapses : public AllSpikingSynapses
{
    public:
        CUDA_CALLABLE AllSTDPSynapses();
        CUDA_CALLABLE virtual ~AllSTDPSynapses();

        static IAllSynapses* Create() { return new AllSTDPSynapses(); }
 
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
         *  Check if the back propagation (notify a spike event to the pre neuron)
         *  is allowed in the synapse class.
         *
         *  @retrun true if the back propagation is allowed.
         */
        virtual bool allowBackPropagation();

        /**
         *  Sets the data for Synapses to input's data.
         *
         *  @param  input  istream to read from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void deserialize(istream& input, IAllNeurons &neurons, const ClusterInfo *clr_info);

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
#endif // USE_GPU

    public:
#if !defined(USE_GPU)
        /*
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         * @param iStep     simulation steps to advance.
         */
        CUDA_CALLABLE virtual void advanceSpikeQueue(int iStep);
#endif // !USE_GPU

        /**
         *  Advance one specific Synapse.
         *  Update the state of synapse for a time step
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
         *  Prepares Synapse for a spike hit (for back propagation).
         *
         *  @param  iSyn   Index of the Synapse to connect to.
         *  @param  iStepOffset  Offset from the current simulation step.
         */
        CUDA_CALLABLE virtual void postSpikeHit(const BGSIZE iSyn, int iStepOffset);
#endif

    protected:
        /**
         *  Checks if there is an input spike in the queue (for back propagation).
         *
         *  @param  iSyn             Index of the Synapse to connect to.
         *  @param  iStepOffset      Offset from the current simulation step.
         *  @param  pISynapsesProps  Pointer to the synapses properties.
         *  @return true if there is an input spike event.
         */
        CUDA_CALLABLE bool isSpikeQueuePost(const BGSIZE iSyn, int iStepOffset, IAllSynapsesProps* pISynapsesProps);

    private:
        /**
         *  Adjust synapse weight according to the Spike-timing-dependent synaptic modification
         *  induced by natural spike trains
         *
         *  @param  iSyn             Index of the synapse to set.
         *  @param  delta            Pre/post synaptic spike interval.
         *  @param  epost            Params for the rule given in Froemke and Dan (2002).
         *  @param  epre             Params for the rule given in Froemke and Dan (2002).
         *  @param  pISynapsesProps  Pointer to the synapses properties.
         */
        CUDA_CALLABLE void stdpLearning(const BGSIZE iSyn,double delta, double epost, double epre, IAllSynapsesProps* pISynapsesProps);
};

#if defined(USE_GPU)

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllSTDPSynapsesDevice(IAllSynapses **pAllSynapses, IAllSynapsesProps *pAllSynapsesProps);

#endif // USE_GPU
