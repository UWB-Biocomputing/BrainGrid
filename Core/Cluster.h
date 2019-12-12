/**
 *      @file Cluster.h
 *
 *      @brief Implementation of Cluster for the spiking neunal networks.
 */

/**
 *
 * @class Cluster Cluster.h "Cluster.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * A cluster is a unit of execution corresponding to a thread, a GPU device, or
 * a computing node, depending on the configuration.
 * The Cluster class maintains and manages classes of objects that make up
 * essential components of the spiking neunal network.
 *    -# IAllNeurons: A class to define a list of partiular type of neurons.
 *    -# IAllSynapses: A class to define a list of partiular type of synapses.
 *
 * \image html bg_data_layout.png
 *
 * The network is composed of 3 superimposed 2-d arrays: neurons, synapses, and
 * summation points.
 *
 * Synapses in the synapse map are located at the coordinates of the neuron
 * from which they receive output.  Each synapse stores a pointer into a
 * summation point.
 *
 * If, during an advance cycle, a neuron \f$A\f$ at coordinates \f$x,y\f$ fires, every synapse
 * which receives output is notified of the spike. Those synapses then hold
 * the spike until their delay period is completed.  At a later advance cycle, once the delay
 * period has been completed, the synapses apply their PSRs (Post-Synaptic-Response) to
 * the summation points.
 * Finally, on the next advance cycle, each neuron \f$B\f$ adds the value stored
 * in their corresponding summation points to their \f$V_m\f$ and resets the summation points to
 * zero.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "Global.h"
#include "IAllNeurons.h"
#include "IAllSynapses.h"
#include "SimulationInfo.h"
#include "Connections.h"
#include "Layout.h"
#include <thread>
#include "Barrier.hpp"

/**
 * cereal
 */
#include <cereal/access.hpp> //for load and construct

class Cluster
{
    public:
        Cluster(IAllNeurons *neurons, IAllSynapses *synapses);
        virtual ~Cluster();

        /**
         * Deserializes internal state from a prior run of the simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         *  @param  input       istream to read from.
         *  @param  sim_info    used as a reference to set info for neurons and synapses.
         *  @param  clr_info    cluster informaion, used as a reference to set info for neurons and synapses.
         */
        //virtual void deserialize(istream& input, const SimulationInfo *sim_info, const ClusterInfo *clr_info);

        /**
         * Serializes internal state for the current simulation.
         * This allows simulations to be continued from a particular point, to be restarted, or to be
         * started from a known state.
         *
         *  @param  output      The filestream to write.
         *  @param  sim_info    used as a reference to set info for neurons and synapses.
         *  @param  clr_info    cluster informaion, used as a reference to set info for neurons and synapses.
         */
        //virtual void serialize(ostream& output, const SimulationInfo *sim_info, const ClusterInfo *clr_info);

        /**
         *  Creates all the Neurons and generates data for them.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      A class to define neurons' layout information in the network.
         *  @param  clr_info    ClusterInfo class to read information from.
         */
        virtual void setupCluster(SimulationInfo *sim_info, Layout *layout, ClusterInfo *clr_info);

        /**
         *  Clean up the cluster.
         *
         *  @param  sim_info    SimulationInfo to refer.
         *  @param  clr_info    ClusterInfo to refer.
         */
        virtual void cleanupCluster(SimulationInfo *sim_info, ClusterInfo *clr_info);

#if defined(USE_GPU)        
        /**
         *  Copy GPU Synapse data back to CPU.
         *
         *  @param  sim_info    SimulationInfo to refer.
         *  @param  clr_info    ClusterInfo to refer.
         */
        virtual void copyGPUSynapseToCPUCluster(SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;
#endif // USE_GPU

        //! Cereal
        template<class Archive>
        static void load_and_construct(Archive& ar, cereal::construct<Cluster>& construct);

        template<class Archive>
        void serialize(Archive & archive);

#if defined(VALIDATION)
        /**
         *  Generates random numbers.
         *
         *  @param  sim_info    SimulationInfo to refer.
         *  @param  clr_info    ClusterInfo to refer.
         */
        virtual void genRandNumbers(const SimulationInfo *sim_info, ClusterInfo *clr_info) = 0;
#endif // VALIDATION

        /**
         * Advances neurons network state of the cluster one simulation step.
         *
         * @param sim_info   parameters defining the simulation to be run with 
         *                   the given collection of neurons.
         * @param clr_info   ClusterInfo to refer.
         * @param iStepOffset  offset from the current simulation step.
         */
        virtual void advanceNeurons(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset) = 0;

        /**
         * Process outgoing spiking data between clusters.
         *
         * @param  clr_info  ClusterInfo to refer.
         */
        virtual void processInterClustesOutgoingSpikes(ClusterInfo *clr_info) = 0;

        /**
         * Process incoming spiking data between clusters.
         *
         * @param  clr_info  ClusterInfo to refer.
         */
        virtual void processInterClustesIncomingSpikes(ClusterInfo *clr_info) = 0;

        /**
         * Advances synapses network state of the cluster one simulation step.
         *
         * @param sim_info - parameters defining the simulation to be run with 
         *                   the given collection of neurons.
         * @param clr_info - ClusterInfo to refer.
         * @param iStepOffset - offset from the current simulation step.
         */
        virtual void advanceSynapses(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset) = 0;

        /**
         * Advances synapses spike event queue state of the cluster.
         *
         * @param sim_info    - SimulationInfo class to read information from.
         * @param clr_info    - ClusterInfo class to read information from.
         * @param iStep       - simulation step to advance.
         */
        virtual void advanceSpikeQueue(const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStep) = 0;

        /**
         *  Create an advanceThread.
         *  If barrier synchronize object has not been created, create it.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  clr_info    ClusterInfo class to read information from.
         *  @param  count       Number of total clusters.
         */
        void createAdvanceThread(const SimulationInfo *sim_info, ClusterInfo *clr_info, int count);

        /**
         *  Thread for advance a cluster.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  clr_info    ClusterInfo class to read information from.
         */
        void advanceThread(const SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         *  Run advance of all waiting threads.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  iStep       Simulation steps to advance.
         */
        static void runAdvance(const SimulationInfo *sim_info, int iStep);

        /**
         *  Quit all advanceThread.
         */
        static void quitAdvanceThread();

    public:
        /**
         *  Pointer to the Neurons object.
         */
        IAllNeurons *m_neurons;

        /**
         *  Pointer to the Synapses object.
         */
        IAllSynapses *m_synapses;

        /**
         *  Pointer to the Synapse Index Map object.
         */
        SynapseIndexMap *m_synapseIndexMap;

    private:
        /**
         *  Pointer to the Barrier Synchnonize object for advanceThreads.
         */
        static Barrier *m_barrierAdvance;

        /**
         *  Flag for advanceThreads. true if terminating advanceThreads.
         */
        static bool m_isAdvanceExit;

        /**
         * The synaptic transmission delay, descretized into time steps
         */
        static int m_nSynapticTransDelay;
};

//! Cereal Serialization/Deserialization Method
template<class Archive>
void Cluster::serialize(Archive & archive) {
    AllSynapses * castm_synapses = dynamic_cast<AllSynapses*>(m_synapses);
    archive(*castm_synapses);
}

//! Cereal Load_and_construct Method
template <class Archive>
void Cluster::load_and_construct( Archive & ar, cereal::construct<Cluster> & construct )
{  
    IAllNeurons *m_neurons2 = nullptr;
    IAllSynapses *m_synapses2 = nullptr;
    AllSynapses * castm_synapses = dynamic_cast<AllSynapses*>(m_synapses2);
    ar(*castm_synapses);
    construct(m_neurons2, m_synapses2);
}