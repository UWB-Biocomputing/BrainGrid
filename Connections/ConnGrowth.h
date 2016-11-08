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
         *  @param  sim_info  SimulationInfo class to read information from.
         *  @param  layout    Layout information of the neunal network.
         *  @param  neurons   The Neuron list to search from.
         *  @param  synapses  The Synapse list to search from.
         */
        virtual void setupConnections(const SimulationInfo *sim_info, Layout *layout, IAllNeurons *neurons, IAllSynapses *synapses);

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
        virtual void deserialize(istream& input, const SimulationInfo *sim_info);

        /**
         *  Writes the intermediate connection status to ostream.
         *
         *  @param  output   ostream to write status to.
         *  @param  sim_info SimulationInfo class to read information from.
         */
        virtual void serialize(ostream& output, const SimulationInfo *sim_info);

        /**
         *  Update the connections status in every epoch.
         *
         *  @param  neurons  The Neuron list to search from.
         *  @param  sim_info SimulationInfo class to read information from.
         *  @param  layout   Layout information of the neunal network.
         *  @return true if successful, false otherwise.
         */
        virtual bool updateConnections(IAllNeurons &neurons, const SimulationInfo *sim_info, Layout *layout);

        /**
         *  Creates a recorder class object for the connection.
         *  This function tries to create either Xml recorder or
         *  Hdf5 recorder based on the extension of the file name.
         *
         *  @param  simInfo              SimulationInfo to refer from.
         *  @return Pointer to the recorder class object.
         */
        virtual IRecorder* createRecorder(const SimulationInfo *sim_info);
#if defined(USE_GPU)
    public:
        /**
         *  Update the weight of the Synapses in the simulation.
         *  Note: Platform Dependent.
         *
         *  @param  num_neurons         number of neurons to update.
         *  @param  neurons             the Neuron list to search from.
         *  @param  synapses            the Synapse list to search from.
         *  @param  sim_info            SimulationInfo to refer from.
         *  @param  m_allNeuronsDevice  Reference to the allNeurons struct on device memory. 
         *  @param  m_allSynapsesDevice Reference to the allSynapses struct on device memory.
         *  @param  layout              Layout information of the neunal network.
         */
        virtual void updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, AllSpikingNeuronsDeviceProperties* m_allNeuronsDevice, AllSpikingSynapsesDeviceProperties* m_allSynapsesDevice, Layout *layout);
#else
    public:
        /**
         *  Update the weight of the Synapses in the simulation.
         *  Note: Platform Dependent.
         *
         *  @param  num_neurons Number of neurons to update.
         *  @param  ineurons    The Neuron list to search from.
         *  @param  isynapses   The Synapse list to search from.
         *  @param  sim_info    SimulationInfo to refer from.
         */
        virtual void updateSynapsesWeights(const int num_neurons, IAllNeurons &neurons, IAllSynapses &synapses, const SimulationInfo *sim_info, Layout *layout);
#endif
    private:
        /**
         *  Calculates firing rates, neuron radii change and assign new values.
         *
         *  @param  neurons  The Neuron list to search from.
         *  @param  sim_info SimulationInfo class to read information from.
         */
        void updateConns(IAllNeurons &neurons, const SimulationInfo *sim_info);

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

};
