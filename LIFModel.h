/**
 * @brief A leaky-integrate-and-fire (I&F) neural network model.
 *
 * @class LIFModel LIFModel.h "LIFModel.h"
 *
 * Implements both neuron and synapse behaviour.
 *
 * A standard leaky-integrate-and-fire neuron model is implemented
 * where the membrane potential \f$V_m\f$ of a neuron is given by
 * \f[
 *   \tau_m \frac{d V_m}{dt} = -(V_m-V_{resting}) + R_m \cdot (I_{syn}(t)+I_{inject}+I_{noise})
 * \f]
 * where \f$\tau_m=C_m\cdot R_m\f$ is the membrane time constant,
 * \f$R_m\f$ is the membrane resistance, \f$I_{syn}(t)\f$ is the
 * current supplied by the synapses, \f$I_{inject}\f$ is a
 * non-specific background current and \f$I_{noise}\f$ is a
 * Gaussian random variable with zero mean and a given variance
 * noise.
 *
 * At time \f$t=0\f$ \f$V_m\f$ is set to \f$V_{init}\f$. If
 * \f$V_m\f$ exceeds the threshold voltage \f$V_{thresh}\f$ it is
 * reset to \f$V_{reset}\f$ and hold there for the length
 * \f$T_{refract}\f$ of the absolute refractory period.
 *
 * The exponential Euler method is used for numerical integration.
 *
 * This model is a rewrite of work by Stiber, Kawasaki, Allan Ortiz, and Cory Mayberry
 *
 * @authors Derek McLean
 */
#pragma once
#ifndef _LIFMODEL_H_
#define _LIFMODEL_H_

#include "Model.h"
#include "Coordinate.h"

#include <vector>

using namespace std;

#define BYTES_OF_DELAYQUEUE         ( sizeof(uint32_t) / sizeof(uint8_t) )
#define LENGTH_OF_DELAYQUEUE        ( BYTES_OF_DELAYQUEUE * 8 )

/**
 * Implementation of Model for the Leaky-Integrate-and-Fire model.
 */
class LIFModel : public Model, TiXmlVisitor
{

    public:
        LIFModel();
        virtual ~LIFModel();

        /*
         * Definitions of concrete implementations of Model interface for an Leaky-Integrate-and-Fire
         * model.
         *
         * @see Model.h
         */

        bool readParameters(TiXmlElement *source);
        void printParameters(ostream &output) const;
        void loadMemory(istream& input, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
        void saveMemory(ostream& output, AllNeurons &neurons, AllSynapses &synapses, FLOAT simulation_step);
        void saveState(ostream& output, const AllNeurons &neurons,  const SimulationInfo &sim_info);
        void createAllNeurons(AllNeurons &neurons, const SimulationInfo &sim_info);
        void setupSim(const int num_neurons, const SimulationInfo &sim_info);
        void advance(AllNeurons& neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
        void updateConnections(const int currentStep, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
        void cleanupSim(AllNeurons &neurons, SimulationInfo &sim_info);
        void logSimStep(const AllNeurons &neurons, const AllSynapses &synapses, const SimulationInfo &sim_info) const;

    protected:

        /* -----------------------------------------------------------------------------------------
         * # Helper Functions
         * ------------------
         */

        // # Read Parameters
        // -----------------

        // Visit an element.
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);
        // Visit an element.
        //bool VisitExit(const TiXmlElement& element);

        // # Print Parameters
        // ------------------

        string neuronToString(AllNeurons& neurons, const int i) const;

        // # Load Memory
        // -------------

        void readNeuron(istream &input, AllNeurons &neurons, const int index);
        void readSynapse(istream &input, AllSynapses &synapses, const int neuron_index, const int synapse_index);
        void initSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index);
        void resetSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index);
        bool updateDecay(AllSynapses &synapses, const int neuron_index, const int synapse_index);

        // # Save Memory
        // -------------

        void writeNeuron(ostream& output, AllNeurons &neurons, const int index) const;
        void writeSynapse(ostream& output, AllSynapses &synapses, const int neuron_index, const int synapse_index) const;

        // # Save State
        // ------------

        void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, const SimulationInfo &sim_info);

        // # Create All Neurons
        // --------------------

        void generateNeuronTypeMap(neuronType neuron_types[], int num_neurons);
        void initStarterMap(bool *starter_map, const int num_neurons, const neuronType neuron_type_map[]);
        void setNeuronDefaults(AllNeurons &neurons, const int index);
        void updateNeuron(AllNeurons &neurons, int neuron_index);

        // # Advance Network/Model
        // -----------------------

        void advanceNeurons(AllNeurons& neurons, AllSynapses &synapses, const SimulationInfo &sim_info);
        void advanceNeuron(AllNeurons& neurons, const int index);
        void fire(AllNeurons &neurons, const int index) const;
        void preSpikeHit(AllSynapses &synapses, const int neuron_index, const int synapse_index);

        void advanceSynapses(const int num_neurons, AllSynapses &synapses);
        void advanceSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index);
        bool isSpikeQueue(AllSynapses &synapses, const int neuron_index, const int synapse_index);

        // # Update Connections
        // --------------------

        void updateHistory(int currentStep, FLOAT stepDuration, AllNeurons &neurons);
        void updateFrontiers(const int num_neurons);
        void updateOverlap(FLOAT num_neurons);
        void updateWeights(const int num_neurons, AllNeurons &neurons, AllSynapses &synapses, const SimulationInfo &sim_info);

        void getSpikeCounts(const AllNeurons &neurons, int *spikeCounts);
        void clearSpikeCounts(AllNeurons &neurons);

        void eraseSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index);
        void addSynapse(AllSynapses &synapses, synapseType type, const int src_neuron, const int dest_neuron, Coordinate &source, Coordinate &dest, FLOAT *sum_point, FLOAT deltaT);
        void createSynapse(AllSynapses &synapses, const int neuron_index, const int synapse_index, Coordinate source, Coordinate dest, FLOAT* sp, FLOAT deltaT, synapseType type);

        // -----------------------------------------------------------------------------------------
        // # Generic Functions for handling synapse types
        // ---------------------------------------------

        synapseType synType(AllNeurons &neurons, Coordinate src_coord, Coordinate dest_coord, const int width);
        synapseType synType(AllNeurons &neurons, const int src_neuron, const int dest_neuron);
        int synSign(const synapseType t);
        synapseType synapseOrdinalToType(const int type_ordinal);

        //

        static const FLOAT SYNAPSE_STRENGTH_ADJUSTMENT;

    private:
        /** State of connections in the network. */
        struct Connections;
        struct GrowthParams
        {
            FLOAT epsilon;
            FLOAT beta;
            FLOAT rho;
            FLOAT targetRate; // Spikes/second
            FLOAT maxRate; // = targetRate / epsilon;
            FLOAT minRadius; // To ensure that even rapidly-firing neurons will connect to
                             // other neurons, when within their RFS.
            FLOAT startRadius; // No need to wait a long time before RFs start to overlap
        };

        static const bool STARTER_FLAG; // = true; // true = use endogenously active neurons in simulation

        FLOAT m_Iinject[2];
        FLOAT m_Inoise[2];
        FLOAT m_Vthresh[2];
        FLOAT m_Vresting[2];
        FLOAT m_Vreset[2];
        FLOAT m_Vinit[2];
        FLOAT m_starter_Vthresh[2];
        FLOAT m_starter_Vreset[2];
        FLOAT m_new_targetRate;

        // Tracks the number of parameters that have been read by read params - kind of a hack to do error handling for read params
        int m_read_params;

        //! True if a fixed layout has been provided
        bool m_fixed_layout;

        vector<int> m_endogenously_active_neuron_list;
        vector<int> m_inhibitory_neuron_layout;

        double m_frac_starter_neurons;
        double m_frac_excititory_neurons;

        GrowthParams m_growth;
        Connections *m_conns;
};

/**
 * Maintains intra-epoch state of connections in the network. This includes history and parameters
 * that inform how new connections are made during growth.
 */
struct LIFModel::Connections
{
        static const string MATRIX_TYPE;
        static const string MATRIX_INIT;

        int *spikeCounts;

        VectorMatrix xloc;
        VectorMatrix yloc;

        //! synapse weight
        CompleteMatrix W;
        //! neuron radii
        VectorMatrix radii;
        //! spiking rate
        VectorMatrix rates;
        //! Inter-neuron distance squared
        CompleteMatrix dist2;
        //! distance between connection frontiers
        CompleteMatrix delta;
        //! the true inter-neuron distance
        CompleteMatrix dist;
        //! areas of overlap
        CompleteMatrix area;
        //! neuron's outgrowth
        VectorMatrix outgrowth;
        //! displacement of neuron radii
        VectorMatrix deltaR;

        // track radii
        CompleteMatrix radiiHistory; // state
        // track firing rate
        CompleteMatrix ratesHistory;
        // burstiness Histogram goes through the
        VectorMatrix burstinessHist;
        // spikes history - history of accumulated spikes count of all neurons (10 ms bin)
        VectorMatrix spikesHistory;

        Connections(const int neuron_count, const FLOAT start_radius, const FLOAT growthStepDuration, const FLOAT maxGrowthSteps);
};

#endif
