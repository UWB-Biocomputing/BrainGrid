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

/** 
 *  05/01/2020
 *  Changed the default weight update rule and all formula constants using the 
 *  independent model (a basic STDP model) and multiplicative model in
 *  Froemke and Dan (2002). Spike-timing-dependent synaptic modification induced by natural spike
 *  trains. Nature 416 (3/2002)
 * 
 *  Independent model:
 *  \f$Delta = t_{post}-t_{pre}\f$ with presynaptic spike at time \f$t_{pre}\f$ and
 *  postsynaptic spike at time \f$t_{post}\f$. Then, the weight update is given by
 *  \f$dw =  Apos * exp(-Delta/taupos)\f$ for \f$Delta > 0\f$, and \f$dw =  Aneg *
 *  exp(-Delta/tauneg)\f$ for \f$Delta < 0\f$. dw is the percentage change in synaptic weight.
 *  (set \f$useFroemkeDanSTDP=false\f$ and \f$mupos=muneg=0\f$ for this basic update rule).
 *  
 *  Multiplicative model:
 *  \f$dw = 1.0 + dw * epre * epost\f$ dw is percent change, so adding 1.0 become the scale ratio
 *  \f$W = W * dw\f$ multiply dw (scale ratio) to the current weight to get the new weight
 *  
 *  Note1:This time we don't use useFroemkeDanSTDP (useFroemkeDanSTDP= false) and mupos and muneg (mupos=muneg=0)
 *  Note2:Based on the FroemkeDan paper, the STDP learning rule only applies to excititory synapses, so we
 *  implement it to have only excititory neurons do STDP weight adjustment 
 */

#pragma once

#include "AllSTDPSynapses.h"
#include "AllDynamicSTDPSynapsesProps.h"

class AllDynamicSTDPSynapses : public AllSTDPSynapses
{
    public:
        CUDA_CALLABLE AllDynamicSTDPSynapses();
        CUDA_CALLABLE virtual ~AllDynamicSTDPSynapses();

        static IAllSynapses* Create() { return new AllDynamicSTDPSynapses(); }
 
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
        CUDA_CALLABLE virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);

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
        CUDA_CALLABLE virtual void createSynapse(const BGSIZE iSyn, int source_index, int dest_index, BGFLOAT* sp, const BGFLOAT deltaT, synapseType type);

#if defined(USE_GPU)
    public:
        /**
         *  Create a AllSynapses class object in device
         *
         *  @param pAllSynapses_d      Device memory address to save the pointer of created AllSynapses object.
         *  @param pAllSynapsesProps_d  Pointer to the synapses properties in device memory.
         */
        virtual void createAllSynapsesInDevice(IAllSynapses** pAllSynapses_d, IAllSynapsesProps *pAllSynapsesProps_d);

#endif // USE_GPU

    protected:
        /**
         *  Calculate the post synapse response after a spike.
         *
         *  @param  iSyn             Index of the synapse to set.
         *  @param  deltaT           Inner simulation step duration.
         *  @param  simulationStep   The current simulation step.
         *  @param  pSpikingSynapsesProps  Pointer to the synapses properties.
         */
        CUDA_CALLABLE virtual void changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, uint64_t simulationStep, AllSpikingSynapsesProps* pSpikingSynapsesProps);
};

#if defined(USE_GPU)

/* -------------------------------------*\
|* # CUDA Global Functions
\* -------------------------------------*/

__global__ void allocAllDynamicSTDPSynapsesDevice(IAllSynapses **pAllSynapses, IAllSynapsesProps *pAllSynapsesProps);

#endif // USE_GPU
