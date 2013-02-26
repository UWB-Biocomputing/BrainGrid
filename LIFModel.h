#ifndef _LIFMODEL_H_
#define _LIFMODEL_H_

#include "global.h"

class LIFModel : public Model {

    public:
        LIFModel(FLOAT Iinject[2], FLOAT Inoise[2],FLOAT Vthresh[2],
            FLOAT Vresting[2], FLOAT Vreset[2], FLOAT Vinit[2],
            FLOAT starter_Vthresh[2],FLOAT starter_Vreset[2],
            FLOAT new_targetRate);
        
        void readParameters();
        
        void createAllNeurons(const FLOAT count, AllNeurons &neurons) const;
        
        void advance(FLOAT neuron_count, AllNeurons &neurons, AllSynapses &synapses);
        
        virtual void updateConnections(Network &network) const =0;
        
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
};

#endif
