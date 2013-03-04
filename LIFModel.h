#ifndef _LIFMODEL_H_
#define _LIFMODEL_H_

#include "Model.h"

#include <vector>

using namespace std;

class LIFModel: public Model, TiXmlVisitor
{

    public:
        LIFModel();

        bool readParameters(TiXmlElement *source);
        void printParameters(ostream &output) const;
        void loadState(istream& input, AllNeurons &neurons, AllSynapses &synapses);
        void saveState(ostream& output, const AllNeurons &neurons,  const SimulationInfo &sim_info);
        void createAllNeurons(const int num_neurons, AllNeurons& neurons, const SimulationInfo &sim_info);
        void setupSim(const int num_neurons, const SimulationInfo &sim_info);
        void advance(AllNeurons& neurons, AllSynapses& synapses);
        void updateConnections(const int currentStep, const int num_neurons, AllSynapses &synapses);

    protected:
        // Visit an element.
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);
        // Visit an element.
        //bool VisitExit(const TiXmlElement& element);

        string neuron_to_string(AllNeurons& neurons, const int i) const;

        void generate_neuron_type_map(neuronType neuron_types[], int num_neurons);
        void init_starter_map(bool *starter_map, const int num_neurons, const neuronType neuron_type_map[]);

        void read_neuron(istream& input, AllNeurons &neurons, const int index);

        void getStarterNeuronMatrix(VectorMatrix& matrix, const bool* starter_map, const SimulationInfo &sim_info);

        void advanceNeurons(int num_neurons, AllNeurons& neurons, AllSynapses& synapses);
        void advanceNeuron(AllNeurons& neurons, const int index);
        void fire(AllNeurons &neurons, const int index);
        void preSpikeHit(AllSynapses &synapses, const int group_index, const int synapse_index);

        void advanceSynapses(const int num_neurons, AllSynapses& synapses);
        void advanceSynapse(AllSynapses& synapses, const int group_index, const int synapse_index);
        bool isSpikeQueue(AllSynapses &synapses, const int group_index, const int synapse_index);

        void updateHistory(int currentStep, FLOAT stepDuration, const int num_neurons);
        void updateFrontiers(const int num_neurons);
        void updateOverlap(FLOAT num_neurons);
        void updateWeights(const int num_neurons, AllSynapses &synapses, SimulationInfo* sim_info);
        void erase_synapse(AllSynapses &synapses, const int group_index, const int start_syn, const int end_syn);

    private:
        struct Connections;
        struct GrowthParams;

        double m_Iinject[2];
        double m_Inoise[2];
        double m_Vthresh[2];
        double m_Vresting[2];
        double m_Vreset[2];
        double m_Vinit[2];
        double m_starter_Vthresh[2];
        double m_starter_Vreset[2];
        double m_new_targetRate;

        //! True if a fixed layout has been provided
        bool m_fixed_layout;

        vector<int> m_endogenously_active_neuron_list;
        vector<int> m_inhibitory_neuron_layout;

        double m_frac_starter_neurons;
        double m_frac_excititory_neurons;

        bool starter_flag = true; // true = use endogenously active neurons in simulation

        // TODO : comment
        int m_read_params;

        GrowthParams m_growth;
        Connections *m_conns;
};

struct LIFModel::Connections
{
        static const string MATRIX_TYPE;
        static const string MATRIX_INIT;

        int *spikeCounts;

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

        const int history_size[2];

        Connections(const int neuron_count, const FLOAT start_radius, const FLOAT growthStepDuration, const FLOAT maxGrowthSteps);
};

struct LIFModel::GrowthParams
{
        double epsilon;
        double beta;
        double rho;
        double targetRate; // Spikes/second
        double maxRate; // = targetRate / epsilon;
        double minRadius; // To ensure that even rapidly-firing neurons will connect to
        // other neurons, when within their RFS.
        double startRadius; // No need to wait a long time before RFs start to overlap
};

#endif
