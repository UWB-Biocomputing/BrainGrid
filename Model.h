#ifndef _MODEL_H_
#define _MODEL_H_

#include "global.h"

#include "AllNeurons.h"
#include "AllSynapses.h"

class Model {

    public:
        
        virtual void readParameters() =0;
        
        virtual void createAllNeurons(FLOAT neuron_count, neuronType *neuron_type_map, bool *endogenously_active_neuron_map, AllNeurons &neurons) const =0;
        
        virtual void advance(FLOAT neuron_count, AllNeurons &neurons, AllSynapses &synapses) =0;
        
        virtual void updateConnections(Network &network) const =0;
};

#endif
