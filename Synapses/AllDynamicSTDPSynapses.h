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
#include "AllDynamicSTDPSynapsesProps.h"

class AllDynamicSTDPSynapses : public AllSTDPSynapses
{
    public:
        AllDynamicSTDPSynapses();
        virtual ~AllDynamicSTDPSynapses();

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
        virtual void resetSynapse(const BGSIZE iSyn, const BGFLOAT deltaT);

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
#else // USE_GPU
    protected:
        /**
         *  Calculate the post synapse response after a spike.
         *
         *  @param  iSyn        Index of the synapse to set.
         *  @param  deltaT      Inner simulation step duration.
         *  @param  iStepOffset      Offset from the current simulation step.
         */
        virtual void changePSR(const BGSIZE iSyn, const BGFLOAT deltaT, int iStepOffset);
#endif // defined(USE_GPU)
};

