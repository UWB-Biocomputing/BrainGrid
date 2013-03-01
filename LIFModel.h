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
        
        void createAllNeurons(const FLOAT count, AllNeurons &neurons) const;
        
        void advance(FLOAT neuron_count, AllNeurons &neurons, AllSynapses &synapses);
        
        void updateConnections(Network &network) const;
        
    protected:
        // Visit an element.
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);
        
        // Visit an element.
        //bool VisitExit(const TiXmlElement& element);
        
        void neuron_to_string(AllNeurons &neurons, const int i) const;
    
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
};

#endif
