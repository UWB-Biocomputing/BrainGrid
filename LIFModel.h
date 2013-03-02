#ifndef _LIFMODEL_H_
#define _LIFMODEL_H_

#include "Model.h"

#include <vector>

using namespace std;

class LIFModel : public Model, TiXmlVisitor {

    public:
        LIFModel();
        
        bool readParameters(TiXmlElement *source);
        
        void printParameters(ostream &output) const;
        
        void createAllNeurons(const FLOAT count, AllNeurons &neurons);
        
        void advance(FLOAT num_neurons, AllNeurons &neurons, AllSynapses &synapses);
        
        void updateConnections();
        
    protected:
        // Visit an element.
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);
        
        // Visit an element.
        //bool VisitExit(const TiXmlElement& element);
        
        string neuron_to_string(AllNeurons &neurons, const int i) const;
        void generate_neuron_type_map(neuronType neuron_types[], int num_neurons);
        void init_starter_map(const int num_neurons, const neuronType neuron_type_map[]);

        void advanceNeurons(FLOAT num_neurons, AllNeurons &neurons, AllSynapses &synapses);
        void advanceSynapses(FLOAT num_neurons, AllSynapses &synapses);

        void advanceNeuron(AllNeurons &neurons, int neuron_index, FLOAT& summationPoint);
        void advanceSynapse(AllSynapses &synapses, int i, int z);
    
    private:
        FLOAT m_Iinject[2];
        FLOAT m_Inoise[2];
        FLOAT m_Vthresh[2];
        FLOAT m_Vresting[2];
        FLOAT m_Vreset[2];
        FLOAT m_Vinit[2];
        FLOAT m_starter_Vthresh[2];
        FLOAT m_starter_Vreset[2];
        FLOAT m_new_targetRate;
        
        //! True if a fixed layout has been provided
        bool m_fixed_layout;
        
        //! The starter existence map (T/F).
        bool* m_endogenously_active_neuron_layout;
        vector<int> m_endogenously_active_neuron_list;
        vector<int> m_inhibitory_neuron_layout;
        
        FLOAT m_frac_starter_neurons;
        FLOAT m_frac_excititory_neurons;
        
        bool starter_flag = true;  // true = use endogenously active neurons in simulation
        
        // TODO : comment
        int m_read_params;

        struct {
        	FLOAT epsilon;
        	FLOAT beta;
        	FLOAT rho;
        	FLOAT targetRate;  // Spikes/second
        	FLOAT maxRate;  // = targetRate / epsilon;
        	FLOAT minRadius;  // To ensure that even rapidly-firing neurons will connect to
        	// other neurons, when within their RFS.
        	FLOAT startRadius;  // No need to wait a long time before RFs start to overlap
        } m_growth;
};

#endif
