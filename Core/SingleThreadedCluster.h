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

/**
 * cereal
 */
//#include <cereal/types/polymorphic.hpp> //for inheritance
//#include <cereal/types/base_class.hpp> //inherit data member from base class
//#include <cereal/access.hpp> //for load and construct

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

        //! Cereal
        /*template<class Archive>
        static void load_and_construct(Archive& ar, cereal::construct<SingleThreadedCluster>& construct);

        template<class Archive>
        void serialize(Archive & archive);*/

#if defined(VALIDATION)
        /**
         *  Generates random numbers.
         *
         *  @param  sim_info    SimulationInfo to refer.
         *  @param  clr_info    ClusterInfo to refer.
         */
        virtual void genRandNumbers(const SimulationInfo *sim_info, ClusterInfo *clr_info);
#endif // VALIDATION

        /**
         * Advances neurons network state of the cluster one simulation step.
         *
         * @param sim_info   parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param clr_info   ClusterInfo to refer.
         * @param iStepOffset  offset from the current simulation step.
         */
        virtual void advanceNeurons(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset);

        /**
         * Process outgoing spiking data between clusters.
         *
         * @param  clr_info  ClusterInfo to refer.
         */
        virtual void processInterClustesOutgoingSpikes(ClusterInfo *clr_info);

        /**
         * Process incoming spiking data between clusters.
         *
         * @param  clr_info  ClusterInfo to refer.
         */
        virtual void processInterClustesIncomingSpikes(ClusterInfo *clr_info);

        /**
         * Advances synapses network state of the cluster one simulation step.
         *
         * @param sim_info   parameters defining the simulation to be run with
         *                   the given collection of neurons.
         * @param clr_info  ClusterInfo to refer.
         * @param iStepOffset  offset from the current simulation step.
         */
        virtual void advanceSynapses(const SimulationInfo *sim_info, ClusterInfo *clr_info, int iStepOffset);

        /**
         * Advances synapses spike event queue state of the cluster one simulation step.
         *
         * @param sim_info    parameters defining the simulation to be run with
         *                    the given collection of neurons.
         * @param clr_info    ClusterInfo to refer.
         * @param iStep       simulation step to advance.
         */
        virtual void advanceSpikeQueue(const SimulationInfo *sim_info, const ClusterInfo *clr_info, int iStep);
};

//! Cereal Serialization/Deserialization Method
/*template<class Archive>
void SingleThreadedCluster::serialize(Archive & archive) { 
        archive(cereal::base_class<Cluster>(this));
}

//! Cereal Load_and_construct Method
template<class Archive>
void SingleThreadedCluster::load_and_construct(Archive& ar, cereal::construct<SingleThreadedCluster>& construct)
{   
        IAllNeurons *m_neurons2 = nullptr;
        IAllSynapses *m_synapses2 = nullptr;
        AllSynapses * castm_synapses = dynamic_cast<AllSynapses*>(m_synapses2);
        ar(*castm_synapses);
        construct(m_neurons2,m_synapses2);
}*/

//! Cereal
//CEREAL_REGISTER_TYPE(SingleThreadedCluster)
