/**
 *      @file AllSTDPSynapsesProperties.h
 *
 *      @brief A container of the base class of all synapse data
 */

#pragma once

#include "AllSpikingSynapsesProperties.h"

class AllSTDPSynapsesProperties : public AllSpikingSynapsesProperties
{
    public:
        AllSTDPSynapsesProperties();
        virtual ~AllSTDPSynapsesProperties();

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  num_neurons   Total number of neurons in the network.
         *  @param  max_synapses  Maximum number of synapses per neuron.
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  clr_info  ClusterInfo class to read information from.
         */
        virtual void setupSynapsesProperties(const int num_neurons, const int max_synapses, SimulationInfo *sim_info, ClusterInfo *clr_info);

    protected:
        /**
         *  Cleanup the class.
         *  Deallocate memories.
         */
        virtual void cleanupSynapsesProperties();

    public:
        /**
         *  The synaptic transmission delay (delay of dendritic backpropagating spike),
         *  descretized into time steps.
         */
        int *total_delayPost;

        /**
         *  Used for extended rule by Froemke and Dan. See Froemke and Dan (2002).
         *  Spike-timing-dependent synaptic modification induced by natural spike trains.
         *  Nature 416 (3/2002).
         */
        BGFLOAT *tauspost;

        /**
         *  sed for extended rule by Froemke and Dan.
         */
        BGFLOAT *tauspre;

        /**
         *  Timeconstant of exponential decay of positive learning window for STDP.
         */
        BGFLOAT *taupos;

        /**
         *  Timeconstant of exponential decay of negative learning window for STDP.
         */
        BGFLOAT *tauneg;

        /**
         *  No learning is performed if \f$|Delta| = |t_{post}-t_{pre}| < STDPgap\f$
         */
        BGFLOAT *STDPgap;
        /**
         *  The maximal/minimal weight of the synapse [readwrite; units=;]
         */
        BGFLOAT *Wex;

        /**
         *  Defines the peak of the negative exponential learning window.
         */
        BGFLOAT *Aneg;

        /**
         *  Defines the peak of the positive exponential learning window.
         */
        BGFLOAT *Apos;

        /**
         *  Extended multiplicative positive update:
         *  \f$dw = (Wex-W)^{mupos} * Apos * exp(-Delta/taupos)\f$.
         *  Set to 0 for basic update. See Guetig, Aharonov, Rotter and Sompolinsky (2003).
         *  Learning input correlations through non-linear asymmetric Hebbian plasticity.
         *  Journal of Neuroscience 23. pp.3697-3714.
         */
        BGFLOAT *mupos;

        /**
         *  Extended multiplicative negative update:
         *  \f$dw = W^{mupos} * Aneg * exp(Delta/tauneg)\f$. Set to 0 for basic update.
         */
        BGFLOAT *muneg;

        /**
         *  True if use the rule given in Froemke and Dan (2002).
         */
        bool *useFroemkeDanSTDP;

        /**
         * The collection of synaptic transmission delay queue.
         */
        EventQueue *postSpikeQueue;
};
