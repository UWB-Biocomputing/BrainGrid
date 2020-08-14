/**
 *      @file ConnStatic.h
 *
 *      @brief The model of the small world network
 */

/**
 *
 * @class ConnStatic ConnStatic.h "ConnStatic.h"
 *
 * \latexonly  \subsubsection*{Implementation} \endlatexonly
 * \htmlonly   <h3>Implementation</h3> \endhtmlonly
 *
 * The small-world networks are regular networks rewired to introduce increasing amounts
 * of disorder, which can be highly clustered, like regular lattices, yet have small
 * characterisic path length, like random graphs. 
 *
 * The structural properties of these graphs are quantified by their characteristic path
 * length \f$L(p)\f$ and clustering coefficient \f$C(p)\f$. Here \f$L\f$ is defined as the number of edges
 * in the shortest path between two vertices, average over all pairs of vertices.
 * The clustering coefficient \f$C(p)\f$ is defined as follows. Suppose that a vertex \f$v\f$ has \f$k_v\f$
 * neighbours; then at most \f$k_v (k_v - 1) / 2\f$ edges can exist between them (this occurs when
 * every neighbour of \f$v\f$ is connected to every other neighbour of \f$v\f$).
 * Let \f$C_v\f$ denote the fracion of these allowable edges that actually exist.
 * Define \f$C\f$ as the avarage of \f$C_v\f$ over all \f$v\f$ (Watts etal. 1998).
 *
 * We first create a regular network characterised by two parameters: number of maximum 
 * connections per neurons and connection radius threshold, then rewire it according 
 * to the small-world rewiring probability.
 *
 * \latexonly  \subsubsection*{Credits} \endlatexonly
 * \htmlonly   <h3>Credits</h3> \endhtmlonly
 *
 * Some models in this simulator is a rewrite of CSIM (2006) and other
 * work (Stiber and Kawasaki (2007?))
 */

#pragma once

#include "Global.h"
#include "Connections.h"
#include "SimulationInfo.h"
#include <vector>
#include <iostream>
#if defined(USE_GPU)
class Barrier;
#endif // USE_GPU

using namespace std;

class ConnStatic : public Connections
{
    public:
        ConnStatic();
        virtual ~ConnStatic();

        static Connections* Create() { return new ConnStatic(); }

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *  Initialize the small world network characterized by parameters: 
         *  number of maximum connections per neurons, connection radius threshold, and
         *  small-world rewiring probability.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  vtClr       Vector of Cluster class objects.
         *  @param  vtClrInfo   Vector of ClusterInfo.
         */
        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);

#if defined(USE_GPU)
        /**
         *  Thread for setting the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  clr         Pointer to cluster class to read information from.
         *  @param  clr_info    Pointer to clusterInfo class to read information from.
         */
        void setupConnectionsThread(const SimulationInfo *sim_info, Layout *layout, Cluster * clr, ClusterInfo * clr_info);
#endif // USE_GPU

        /**
         *  Cleanup the class.
         */
        virtual void cleanupConnections();

        /**
         *  Checks the number of required parameters to read.
         *
         * @return true if all required parameters were successfully read, false otherwise.
         */
        virtual bool checkNumParameters();

        /**
         *  Attempts to read parameters from a XML file.
         *
         *  @param  element TiXmlElement to examine.
         *  @return true if successful, false otherwise.
         */
        virtual bool readParameters(const TiXmlElement& element);

        /**
         *  Prints out all parameters of the connections to ostream.
         *
         *  @param  output  ostream to send output to.
         */
        virtual void printParameters(ostream &output) const;

        /**
         *  Reads the intermediate connection status from istream.
         *
         *  @param  input    istream to read status from.
         *  @param  sim_info SimulationInfo class to read information from.
         */
        virtual void deserialize(istream& input, const SimulationInfo *sim_info);

        /**
         *  Writes the intermediate connection status to ostream.
         *
         *  @param  output   ostream to write status to.
         *  @param  sim_info SimulationInfo class to read information from.
         */
        virtual void serialize(ostream& output, const SimulationInfo *sim_info);

        /**
         *  Creates a recorder class object for the connection.
         *  This function tries to create either Xml recorder or
         *  Hdf5 recorder based on the extension of the file name.
         *
         *  @param  simInfo              SimulationInfo to refer from.
         *  @return Pointer to the recorder class object.
         */
        virtual IRecorder* createRecorder(const SimulationInfo *sim_info);

#if defined(BOOST_PYTHON)
    public:
#else // BOOSTPYTHON
    private:
#endif // BOOSTPYTHON
        //! number of maximum connections per neurons
        int m_nConnsPerNeuron;

        //! Connection radius threshold
        BGFLOAT m_threshConnsRadius;

        //! Small-world rewiring probability
        BGFLOAT m_pRewiring;

        //! Min/max values of excitatory neuron's synapse weight
        BGFLOAT m_excWeight[2];

        //! Min/max values of inhibitory neuron's synapse weight
        BGFLOAT m_inhWeight[2];

#if defined(USE_GPU)
        //! Barrier Synchnonize object for setupConnections
        static Barrier *m_barrierSetupConnections;
#endif // USE_GPU
 
public:
        struct DistDestNeuron
        {
            BGFLOAT dist;     // destance to the destination neuron
            int src_neuron;  // index of the destination neuron

#if defined(USE_GPU) && defined(__CUDACC__)
            __device__ __host__
#endif // USE_GPU
            bool operator<(const DistDestNeuron& other) const
            {
                return (dist < other.dist);
            }
        };
};

#if defined(USE_GPU) && defined(__CUDACC__)
/**
 *  CUDA kernel function for setting up connections.
 *  Setup the internal structure of the class (allocate memories and initialize them).
 *  Initialize the small world network characterized by parameters:
 *  number of maximum connections per neurons, connection radius threshold, and
 *  small-world rewiring probability.
 *
 *  @param] synapsesDevice      Pointer to the Synapses object in device memory.
 *  @param  num_neurons         Number of total neurons.
 *  @param  totalClusterNeurons Total number of neurons in the cluster.
 *  @param  clusterNeuronsBegin Begin neuron index of the cluster.
 *  @param  xloc_d              Pointer to the neuron's x location array.
 *  @param  yloc_d              Pointer to the neuron's y location array.
 *  @param  nConnsPerNeuron     Number of maximum connections per neurons.
 *  @param  threshConnsRadius   Connection radius threshold.
 *  @param  neuron_type_map_d   Pointer to the neurons type map in device memory.
 *  @param  rDistDestNeuron_d   Pointer to the DistDestNeuron structure array.
 *  @param  deltaT              The time step size.
 *  @param  allNeuronsProps     Pointer to the Neuron structures in device memory.
 *  @param  allSynapsesProps    Pointer to the Synapse structures in device memory.
 *  @param  minExcWeight        Min values of excitatory neuron's synapse weight.
 *  @param  maxExcWeight        Max values of excitatory neuron's synapse weight.
 *  @param  minInhWeight        Min values of inhibitory neuron's synapse weight.
 *  @param  maxInhWeight        Max values of inhibitory neuron's synapse weight.
 *  @param  devStates_d         Curand global state.
 *  @param  seed                Seed for curand.
 */
extern __global__ void setupConnectionsDevice( IAllSynapses* synapsesDevice, int num_neurons, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* xloc_d, BGFLOAT* yloc_d, int nConnsPerNeuron, int threshConnsRadius, neuronType* neuron_type_map_d, ConnStatic::DistDestNeuron *rDistDestNeuron_d, BGFLOAT deltaT, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, BGFLOAT minExcWeight, BGFLOAT maxExcWeight, BGFLOAT minInhWeight, BGFLOAT maxInhWeight, curandState* devStates_d, unsigned long seed );

#endif // USE_GPU
