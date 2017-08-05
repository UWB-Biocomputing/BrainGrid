/**
 *      @file SingleThreadedCluster.h
 *
 *      @brief Implementation of Cluster for the spiking neunal networks.
 */

/**
 *
 * @class SingleThreadedCluster SingleThreadedCluster.h "SingleThreadedCluster.h"
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

#include "Cluster.h"

class SingleThreadedCluster : public Cluster {
    public:
        // Constructor & Destructor
        SingleThreadedCluster(IAllNeurons *neurons, IAllSynapses *synapses);
        ~SingleThreadedCluster();

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

        /**
         * Advances neurons network state of the cluster one simulation step.
         *
         * @param sim_info   parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param  clr_info   ClusterInfo to refer.
         */
        virtual void advanceNeurons(const SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         * Advances synapses network state of the cluster one simulation step.
         *
         * @param sim_info   parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param  clr_info  ClusterInfo to refer.
         */
        virtual void advanceSynapses(const SimulationInfo *sim_info, ClusterInfo *clr_info);

        /**
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         * @param  clr_info    ClusterInfo to refer.
         */
        virtual void advanceSpikeQueue(const ClusterInfo *clr_info);
};
