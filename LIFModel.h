#ifndef _LIFMODEL_H_
#define _LIFMODEL_H_

#include "Model.h"

class LIFModel : public Model, TiXmlVisitor {

    public:
        LIFModel();
        
        bool readParameters(TiXmlElement *source);
        
        void createAllNeurons(const FLOAT count, AllNeurons &neurons) const;
        
        void advance(FLOAT neuron_count, AllNeurons &neurons, AllSynapses &synapses);
        
        void updateConnections(Network &network) const;
        
    protected:
        // Visit an element.
        bool VisitEnter(const TiXmlElement& element, const TiXmlAttribute* firstAttribute);
        
        // Visit an element.
        bool VisitExit(const TiXmlElement& element);
    
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
        
        // TODO : comment
        int m_read_params;
};

#endif
