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
#include <iostream>

using namespace std;

/**
 * Implementation of Model for the Leaky-Integrate-and-Fire model.
 */
class LIFModel : public Model, TiXmlVisitor
{

    public:
        LIFModel();
        virtual ~LIFModel();

        /*
         * Declarations of concrete implementations of Model interface for an Leaky-Integrate-and-Fire
         * model.
         *
         * @see Model.h
         */
		bool initializeModel(SimulationInfo *sim_info, AllNeurons& neurons, AllSynapses& synapses);
        bool readParameters(TiXmlElement *source);
        void printParameters(ostream &output) const;
        void loadMemory(istream& input, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info);
        void saveMemory(ostream& output, AllNeurons &neurons, AllSynapses &synapses, BGFLOAT simulation_step);
        void saveState(ostream& output, const AllNeurons &neurons,  SimulationInfo *sim_info);
        void createAllNeurons(AllNeurons &neurons, SimulationInfo *sim_info);
        void setupSim(const uint32_t num_neurons, SimulationInfo *sim_info);
        void advance(AllNeurons& neurons, AllSynapses &synapses, SimulationInfo *sim_info);
        void cleanupSim(AllNeurons &neurons, SimulationInfo *sim_info);
        void logSimStep(const AllNeurons &neurons, const AllSynapses &synapses, SimulationInfo *sim_info) const;

    protected:

        /* -----------------------------------------------------------------------------------------
         * # Helper Functions
         * ------------------
         */

        // # Read Parameters
        // -----------------

        // Parse an element for parameter values.
        // Required by TiXmlVisitor, which is used by #readParameters
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);
		/// Visit a document.
		bool VisitEnter( const TiXmlDocument& doc )	{ return TiXmlVisitor::VisitEnter(doc); }
		/// Visit a document.
		bool VisitExit( const TiXmlDocument& doc )	{ return TiXmlVisitor::VisitExit(doc); }
		/// Visit an element.
		bool VisitExit( const TiXmlElement& element )			{ return TiXmlVisitor::VisitExit(element); }
		/// Visit a declaration
		bool Visit( const TiXmlDeclaration& declaration )		{ return TiXmlVisitor::Visit(declaration); }
		/// Visit a text node
		bool Visit( const TiXmlText& text )						{ return TiXmlVisitor::Visit(text); }
		/// Visit a comment node
		bool Visit( const TiXmlComment& comment )				{ return TiXmlVisitor::Visit(comment); }
		/// Visit an unknown node
		bool Visit( const TiXmlUnknown& unknown )				{ return TiXmlVisitor::Visit(unknown); }

        // # Print Parameters
        // ------------------

        // Constructs a string representation of a specific neuron in the network.
        string neuronToString(AllNeurons& neurons, const uint32_t i) const;

        // # Load Memory
        // -------------

        // Deserialize a neuron from some input source.
        void readNeuron(istream &input, AllNeurons &neurons, const uint32_t index);
        // Deserialize a synapse from some input source.
        void readSynapse(istream &input, AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);
        // TODO
        void initSpikeQueue(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);
        // TODO
        void resetSynapse(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);
        // TODO
        bool updateDecay(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);

        // # Save Memory
        // -------------

        // Serialize a neuron to an output destination
        void writeNeuron(ostream& output, AllNeurons &neurons, const uint32_t index) const;
        // Serialize a synapse to an output destination
        void writeSynapse(ostream& output, AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index) const;

        // # Save State
        // ------------

        // TODO
        void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, SimulationInfo *sim_info);

        // # Create All Neurons
        // --------------------

        // TODO
        void generateNeuronTypeMap(vector<neuronType> &neuron_types, uint32_t num_neurons);
        // TODO
        void initStarterMap(bool *starter_map, const uint32_t num_neurons, const vector<neuronType> &neuron_type_map);
        // TODO
        void setNeuronDefaults(AllNeurons &neurons, const uint32_t index);
        // TODO
        void updateNeuron(AllNeurons &neurons, uint32_t neuron_index);

        // # Advance Network/Model
        // -----------------------

        // Update the state of all neurons for a time step
        void advanceNeurons(AllNeurons& neurons, AllSynapses &synapses, SimulationInfo *sim_info);
        // Helper for #advanceNeuron. Updates state of a single neuron.
        void advanceNeuron(AllNeurons& neurons, const uint32_t index);
        // Initiates a firing of a neuron to connected neurons
        void fire(AllNeurons &neurons, const uint32_t index) const;
        // TODO
        void preSpikeHit(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);

        // Update the state of all synapses for a time step
        void advanceSynapses(const uint32_t num_neurons, AllSynapses &synapses);
        // Helper for #advanceSynapses. Updates state of a single synapse.
        void advanceSynapse(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);
        // TODO
        bool isSpikeQueue(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);

        // # Update Connections
        // --------------------

        void updateHistory(uint32_t currentStep, BGFLOAT epochDuration, AllNeurons &neurons);
        // TODO
        void updateFrontiers(const uint32_t num_neurons);
        // TODO
        void updateOverlap(BGFLOAT num_neurons);
        // TODO
        void updateWeights(const uint32_t num_neurons, AllNeurons &neurons, AllSynapses &synapses, SimulationInfo *sim_info);

        // TODO
        void getSpikeCounts(const AllNeurons &neurons);
        // TODO
        void clearSpikeCounts(AllNeurons &neurons);

        // TODO
        void eraseSynapse(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index);
        // TODO
        void addSynapse(AllSynapses &synapses, synapseType type, const uint32_t src_neuron, const uint32_t dest_neuron, Coordinate &source, Coordinate &dest, uint32_t sum_point, TIMEFLOAT deltaT);
        // TODO
        void createSynapse(AllSynapses &synapses, const uint32_t neuron_index, const uint32_t synapse_index, Coordinate source, Coordinate dest, uint32_t sp, TIMEFLOAT deltaT, synapseType type);

        // -----------------------------------------------------------------------------------------
        // # Generic Functions for handling synapse types
        // ---------------------------------------------

        // Determines the type of synapse for a synapse at a given location in the network.
        synapseType synType(AllNeurons &neurons, Coordinate src_coord, Coordinate dest_coord, const uint32_t width);
        // Determines the type of synapse for a synapse between two neurons.
        synapseType synType(AllNeurons &neurons, const uint32_t src_neuron, const uint32_t dest_neuron);
        // Determines the direction of the weight for a given synapse type.
        int32_t synSign(const synapseType t);
        // Converts the ordinal representation of a synapse type to its enum value.
        synapseType synapseOrdinalToType(const uint32_t type_ordinal);

        // TODO
        static const BGFLOAT SYNAPSE_STRENGTH_ADJUSTMENT;

        /** State of connections in the network. (must be accessible to CUDA_LIFModel) */
        struct Connections;
        // TODO
        Connections *m_conns;
    private:

        // TODO
        struct GrowthParams
        {
            // TODO
            BGFLOAT epsilon;
            // TODO
            BGFLOAT beta;
            // TODO
            BGFLOAT rho;
            BGFLOAT targetRate; // Spikes/second
            BGFLOAT maxRate; // = targetRate / epsilon;
            BGFLOAT minRadius; // To ensure that even rapidly-firing neurons will connect to
                             // other neurons, when within their RFS.
            BGFLOAT startRadius; // No need to wait a long time before RFs start to overlap
        };

        // TODO
        friend ostream& operator<<(ostream &out, const GrowthParams &params);

        static const bool STARTER_FLAG; // = true; // true = use endogenously active neurons in simulation

        // TODO
        BGFLOAT m_Iinject[2];
        // TODO
        BGFLOAT m_Inoise[2];
        // TODO
        BGFLOAT m_Vthresh[2];
        // TODO
        BGFLOAT m_Vresting[2];
        // TODO
        BGFLOAT m_Vreset[2];
        // TODO
        BGFLOAT m_Vinit[2];
        // TODO
        BGFLOAT m_starter_Vthresh[2];
        // TODO
        BGFLOAT m_starter_Vreset[2];
        // TODO
        BGFLOAT m_new_targetRate;

        // Tracks the number of parameters that have been read by read params - kind of a hack to do error handling for read params
        uint32_t m_read_params;

        //! True if a fixed layout has been provided
        bool m_fixed_layout;

        // TODO
        vector<uint32_t> m_endogenously_active_neuron_list;
        // TODO
        vector<uint32_t> m_inhibitory_neuron_layout;

        // TODO
        BGFLOAT m_frac_starter_neurons;
        // TODO
        BGFLOAT m_frac_excititory_neurons;

        // TODO
        GrowthParams m_growth;
};

/**
 * Maintains intra-epoch state of connections in the network. This includes history and parameters
 * that inform how new connections are made during growth.
 */
struct LIFModel::Connections
{
        // TODO
        static const string MATRIX_TYPE;
        // TODO
        static const string MATRIX_INIT;

        // TODO
        uint32_t *spikeCounts;

        // TODO
        VectorMatrix xloc;
        // TODO
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

        // TODO
        Connections(const uint32_t neuron_count, const BGFLOAT start_radius, const BGFLOAT growthepochDuration, const BGFLOAT maxGrowthSteps);
};

#endif
