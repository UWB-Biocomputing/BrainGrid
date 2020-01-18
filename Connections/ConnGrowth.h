/**
 *      @file ConnGrowth.h
 *
 *      @brief The model of the activity dependent neurite outgrowth
 */

/**
 *
 * @class ConnGrowth ConnGrowth.h "ConnGrowth.h"
 *
 ** \latexonly  \subsubsection*{Implementation} \endlatexonly
 ** \htmlonly   <h3>Implementation</h3> \endhtmlonly
 **
 * The activity dependent neurite outgrowth model is a phenomenological model derived by
 * a number of studies that demonstarated low level of electric activity (low firing rate)
 * stimulated neurite outgrowth, and high level of electric activity (high firing rate)
 * lead to regression (Ooyen etal. 1995).
 *
 * In this, synaptic strength (connectivity), \f$W\f$, was determined dynamically by a model of neurite
 * (cell input and output region) growth and synapse formation,
 * and a cell's region of connectivity is modeled as a circle with radius that changes
 * at a rate inversely proportional to a sigmoidal function of cell firing rate:
 * \f[
 *  \frac{d R_{i}}{dt} = \rho G(F_{i})
 * \f]
 * \f[
 *  G(F_{i}) = 1 - \frac{2}{1 + exp((\epsilon - F_{i}) / \beta)}
 * \f]
 * where \f$R_{i}\f$ is the radius of connectivity of neuron \f$i\f$, \f$F_{i}\f$ is neuron i's firing rate
 * (normalized to be in the range \f$[0,1]\f$, \f$\rho\f$ is an outgrowth rate constant, \f$\epsilon\f$ is a constant
 * that sets the "null point" for outgrowth (the firing rate in spikes/sec that causes
 * no outgrowth or retration), and \f$\beta\f$ determines the slope of \f$G(\cdot)\f$.
 * One divergence in these simulations from strict modeling of the living preparation
 * was that \f$\rho\f$ was increased to reduce simulated development times from the weeks
 * that the living preparation takes to 60,000s (approximaely 16 simulated hours).
 * Extensive analysis and simulation was performed to determine the maximum \f$\rho\f$ \f$(\rho=0.0001)\f$
 * that would not interfere with network dynamics (the increased value of \f$\rho\f$ was still
 * orders of magnitude slower than the slowest of the neuron or synapse time constants,
 * which were order of \f$10^{-2}\f$~\f$10^{-3}sec\f$).
 *
 * Synaptic strengths were computed for all pairs of neurons that had overlapping connectivity
 * regions as the area of their circle's overlap:
 * \f[
 *  r_0^2 = r_1^2 + |AB|^2 - 2 r_1 |AB| cos(\angle CBA)
 * \f]
 * \f[
 *  cos(\angle CBA) = \frac{r_1^2 + |AB|^2 - r_0^2}{2 r_1 |AB|}
 * \f]
 * \f[
 *  \angle CBD =  2 \angle CBA
 * \f]
 * \f[
 *  cos(\angle CAB) = \frac{r_0^2 + |AB|^2 - r_1^2}{2 r_0 |AB|}
 * \f]
 * \f[
 *  \angle CAD =  2 \angle CAB
 * \f]
 * \f[
 *  w_{01} = \frac{1}{2} \angle CBD r_1^2 - \frac{1}{2} r_1^2 sin(\angle CBD) + \frac{1}{2} \angle CAD r_0^2 - \frac{1}{2} r_0^2 sin(\angle CAD)
 * \f]
 * \f[
 *  w_{01} = w_{10}
 * \f]
 * where A and B are the locations of neurons A and B, \f$r_0\f$ and 
 * \f$r_1\f$ are the neurite radii of neuron A and B, C and B are locations of intersections 
 * of neurite boundaries of neuron A and B, and \f$w_{01}\f$ and \f$w_{10}\f$ are the areas of 
 * their circla's overlap. 
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

/**
 * cereal
 */
#include <cereal/types/polymorphic.hpp> //for inheritance
#include <cereal/types/vector.hpp>

using namespace std;

class ConnGrowth : public Connections
{
    public:
        ConnGrowth();
        virtual ~ConnGrowth();

        static Connections* Create() { return new ConnGrowth(); }

        /**
         *  Setup the internal structure of the class (allocate memories and initialize them).
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  vtClr       Vector of Cluster class objects.
         *  @param  vtClrInfo   Vector of ClusterInfo.
         */
        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);

        /**
         *  Cleanup the class (deallocate memories).
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
        //virtual void deserialize(istream& input, const SimulationInfo *sim_info);

        /**
         *  Writes the intermediate connection status to ostream.
         *
         *  @param  output   ostream to write status to.
         *  @param  sim_info SimulationInfo class to read information from.
         */
        //virtual void serialize(ostream& output, const SimulationInfo *sim_info);

        /**
         *  Update the connections status in every epoch.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  vtClr       Vector of Cluster class objects.
         *  @param  vtClrInfo   Vector of ClusterInfo.
         */
        virtual void updateConnections(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);

        /**
         *  Creates a recorder class object for the connection.
         *  This function tries to create either Xml recorder or
         *  Hdf5 recorder based on the extension of the file name.
         *
         *  @param  simInfo              SimulationInfo to refer from.
         *  @return Pointer to the recorder class object.
         */
        virtual IRecorder* createRecorder(const SimulationInfo *sim_info);

        //! Cereal
        //template<class Archive>
        //void serialize(Archive & archive);

        template<class Archive>
        void save(Archive & archive) const;

        template<class Archive>
        void load(Archive & archive);  

#if defined(USE_GPU)
        void printRadii() const;
#endif     

    private:
        /**
         *  Update the weight of the Synapses in the simulation.
         *  Creates a thread for each cluster and transfer the task.
         *
         *  @param  sim_info    SimulationInfo to refer from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  vtClr       Vector of Cluster class objects.
         *  @param  vtClrInfo   Vector of ClusterInfo.
         */
        void updateSynapsesWeights(const SimulationInfo *sim_info, Layout *layout, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);
  
#if defined(USE_GPU)
        /**
         *  Thread for Updating the weight of the Synapses in the simulation.
         *  Executes a CUDA kernel function to do the task.
         *
         *  @param  sim_info    SimulationInfo to refer from.
         *  @param  layout      Layout information of the neunal network.
         *  @param  clr         Pointer to cluster class to read information from.
         *  @param  clr_info    Pointer to clusterInfo class to read information from.
         */
        void updateSynapsesWeightsThread(const SimulationInfo *sim_info, Layout *layout, Cluster *clr, ClusterInfo *clr_info);
#endif

        /**
         *  Calculates firing rates, neuron radii change and assign new values.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  vtClr       Vector of Cluster class objects.
         *  @param  vtClrInfo   Vector of ClusterInfo.
         */
        void updateConns(const SimulationInfo *sim_info, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo);

#if defined(USE_GPU)
        /**
         *  Thread for calculating firing rates, neuron radii change and assign new values.
         *  Executes a CUDA kernel function to do the task.
         *
         *  @param  sim_info    SimulationInfo class to read information from.
         *  @param  clr         Pointer to cluster class to read information from.
         *  @param  clr_info    Pointer to clusterInfo class to read information from.
         */
        void updateConnsThread(const SimulationInfo *sim_info, Cluster *clr, ClusterInfo *clr_info);
#endif

#if !defined(USE_GPU)
        /**
         *  Update the distance between frontiers of Neurons.
         *
         *  @param  num_neurons Number of neurons to update.
         *  @param  layout      Layout information of the neunal network.
         */
        void updateFrontiers(const int num_neurons, Layout *layout);

        /**
         *  Update the areas of overlap in between Neurons.
         *
         *  @param  num_neurons Number of Neurons to update.
         *  @param  layout      Layout information of the neunal network.
         */
        void updateOverlap(BGFLOAT num_neurons, Layout *layout);
#endif // !USE_GPU

    public:
        struct GrowthParams
        {
            BGFLOAT epsilon;   //null firing rate(zero outgrowth)
            BGFLOAT beta;      //sensitivity of outgrowth to firing rate
            BGFLOAT rho;       //outgrowth rate constant
            BGFLOAT targetRate; // Spikes/second
            BGFLOAT maxRate;   // = targetRate / epsilon;
            BGFLOAT minRadius; // To ensure that even rapidly-firing neurons will connect to
                               // other neurons, when within their RFS.
            BGFLOAT startRadius; // No need to wait a long time before RFs start to overlap
        };

        //! structure to keep growth parameters
        GrowthParams m_growth;

        //! spike count for each epoch
        int *spikeCounts;

#if defined(USE_GPU)
        //! neuron radii
        BGFLOAT *radii;

        //! spiking rate
        BGFLOAT *rates;

        int size;
#else // !USE_GPU
        //! synapse weight
        CompleteMatrix *W;

        //! neuron radii
        VectorMatrix *radii;

        //! spiking rate
        VectorMatrix *rates;

        //! distance between connection frontiers
        CompleteMatrix *delta;

        //! areas of overlap
        CompleteMatrix *area;

        //! neuron's outgrowth
        VectorMatrix *outgrowth;

        //! displacement of neuron radii
        VectorMatrix *deltaR;

#endif // !USE_GPU

private:
#if defined(USE_GPU)
        //! Barrier Synchnonize object for updateConnections
        static Barrier *m_barrierUpdateConnections;
#endif // USE_GPU
};

#if defined(USE_GPU) && defined(__CUDACC__)
/**
 *  CUDA kernel function for calculating firing rates, neuron radii change and assign new values.
 *
 *  @param  allNeuronsDevice       Pointer to Neuron structures in device memory.
 *  @param  totalClusterNeurons    Number of neurons in the cluster.
 *  @param  max_spikes             Maximum firing rate.
 *  @param  epochDuration          One epoch duration in second.
 *  @param  maxRate                Growth parameter (= targetRate / epsilon)
 *  @param  beta                   Growth parameter (sensitivity of outgrowth to firing rate)
 *  @param  rho                    Growth parameter (outgrowth rate constant)
 *  @param  epsilona               Growth parameter (null firing rate(zero outgrowth))
 *  @param  rates_d                Pointer to rates data array.
 *  @param  radii_d                Pointer to radii data array.
 */
extern __global__ void updateConnsDevice( AllSpikingNeuronsProps* allNeuronsProps, int totalClusterNeurons, int max_spikes, BGFLOAT epochDuration, BGFLOAT maxRate, BGFLOAT beta, BGFLOAT rho, BGFLOAT epsilon, BGFLOAT* rates_d, BGFLOAT* radii_d );

/**
 * Adjust the strength of the synapse or remove it from the synapse map if it has gone below
 * zero.
 *
 * @param[in] synapsesDevice     Pointer to the Synapses object in device memory.
 * @param[in] num_neurons        Number of neurons.
 * @param[in] deltaT             The time step size.
 * @param[in] maxSynapses        Maximum number of synapses per neuron.
 * @param[in] allNeuronsProps    Pointer to the Neuron structures in device memory.
 * @param[in] allSynapsesProps   Pointer to the Synapse structures in device memory.
 * @param[in] neuron_type_map_d  Pointer to the neurons type map in device memory.
 * @param[in] totalClusterNeurons  Total number of neurons in the cluster.
 * @param[in] clusterNeuronsBegin  Begin neuron index of the cluster.
 * @param[in] radii_d            Pointer to the rates data array.
 * @param[in] xloc_d             Pointer to the neuron's x location array.
 * @param[in] yloc_d             Pointer to the neuron's y location array.
 */
extern __global__ void updateSynapsesWeightsDevice( IAllSynapses* synapsesDevice, int num_neurons, BGFLOAT deltaT, int maxSynapses, AllSpikingNeuronsProps* allNeuronsProps, AllSpikingSynapsesProps* allSynapsesProps, neuronType* neuron_type_map_d, int totalClusterNeurons, int clusterNeuronsBegin, BGFLOAT* radii_d, BGFLOAT* xloc_d,  BGFLOAT* yloc_d );

#endif // USE_GPU && __CUDACC__

//! Cereal Serialization/Deserialization Method
template<class Archive>
void ConnGrowth::save(Archive & archive) const {
#if defined(USE_GPU)
    /*vector<BGFLOAT> radiiVector;
    for(int i = 0; i < size; i++) {
        radiiVector.push_back(radii[i]);
    }*/
    //archive(radiiVector);
    int a = 5;
    archive(a);
#else  
    int a = 5;
    archive(a);      
    //archive(*radii);
#endif 
}

template<class Archive>
void ConnGrowth::load(Archive & archive) {
#if defined(USE_GPU)
    /*vector<BGFLOAT> radiiVector;
    archive(radiiVector);
    for(int i = 0; i < size; i++) {
        radii[i] = radiiVector[i];
    }*/
    int a = 5;
    archive(a);
#else        
    //archive(*radii);
    int a = 5;
    archive(a);
#endif 
}

//! Cereal
CEREAL_REGISTER_TYPE(ConnGrowth)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Connections,ConnGrowth)
